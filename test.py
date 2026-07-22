"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.TestDataLoader import TestIRSeqDataLoader
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
from numpy import *
import time
from PIL import Image
import cv2
import scipy.io as scio
from ShootingRules import ShootingRules
from sklearn.metrics import auc
from collections import OrderedDict
from thop import profile, clever_format
from tools_forSatVideoIRSTD.seg2centroid_txt import (
    calculate_centroids,
    format_centroid_line,
)
# from attribution.core import IR_Integrated_gradient, MeanLinearPath, ZeroLinearPath
from write_results import writeNUDTMIRSDT_ROC, writeMIRST_ROC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'networks/models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Sequence-window batch size; currently must be 1 [default: 1]')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch of generator to test [default: None]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--seqlen', type=int, default=40, help='Frame number as an input [default: 100]')
    parser.add_argument('--datapath', type=str, default='./datasets/NUDT-MIRSDT', help='Data path')
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT', help='dataset name [default: NUDT-MIRSDT, IRDST-simulation, RGB-T, SatVideoIRSDT]')
    parser.add_argument('--log_dir', type=str, default='NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40', help='experiment root')
    parser.add_argument('--logpath', type=str, default='./log/', help='Log path: ./log/')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--visual_count', type=int, default=0,
                        help='Randomly save this many prediction masks; 0 saves all when --visual is set')
    parser.add_argument('--visual_seed', type=int, default=46,
                        help='Random seed used by --visual_count')
    parser.add_argument('--centroid_txt', action='store_true', default=False,
                        help='Write one centroid TXT file for every validation sequence')
    parser.add_argument('--centroid_threshold', type=float, default=0.5,
                        help='Probability threshold used to produce centroid TXT files')
    parser.add_argument('--centroid_dir', type=str, default=None,
                        help='Centroid TXT output directory; defaults to <experiment>/out_centroid')
    parser.add_argument('--output_only', action='store_true', default=False,
                        help='Skip IoU, Pd/Fa and FLOPs; only create requested output files')
    parser.add_argument('--test_workers', type=int, default=2,
                        help='Persistent workers used to read test windows [default: 2]')
    parser.add_argument('--prefetch_factor', type=int, default=1,
                        help='Test windows prefetched by each worker [default: 1]')
    parser.add_argument('--profile_flops', action='store_true', default=False,
                        help='Run the extra THOP forward pass after evaluation')
    parser.add_argument('--threshold_eval', type=float, default=0.5, help='Threshold in evaluation [default: 0.5]')
    parser.add_argument('--attribution', action='store_true', default=False, help='This test is attribution analysis or not')
    return parser.parse_args()


def count_parameters(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1000 ** 2)  # MB (十进制)


def test_loader_options(worker_count, prefetch_factor):
    options = {'num_workers': worker_count}
    if worker_count > 0:
        options.update({
            'persistent_workers': True,
            'prefetch_factor': prefetch_factor,
        })
    return options


def sequence_storage_length(seq_dataset, seq_len):
    last_frame = max(
        frame_data[2]
        for sample in seq_dataset.samplelist
        for frame_data in sample
    )
    return max(seq_len, last_frame + 1)


def merge_prediction_window(sequence_prediction, window_prediction, start_frame):
    """Merge a window using the original overlap-wise maximum rule."""
    stop_frame = min(
        sequence_prediction.size(1),
        start_frame + window_prediction.size(1),
    )
    window_length = stop_frame - start_frame
    destination = sequence_prediction[:, start_frame:stop_frame]
    torch.maximum(
        destination,
        window_prediction[:, :window_length],
        out=destination,
    )


def copy_annotation_window(sequence_annotation, window_annotation, start_frame):
    stop_frame = min(
        sequence_annotation.size(1),
        start_frame + window_annotation.size(1),
    )
    window_length = stop_frame - start_frame
    sequence_annotation[:, start_frame:stop_frame].copy_(
        window_annotation[:, :window_length]
    )


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    if args.visual_count < 0:
        raise ValueError('visual_count must be non-negative.')
    if not 0 <= args.centroid_threshold <= 1:
        raise ValueError('centroid_threshold must be between 0 and 1.')
    if args.batch_size != 1:
        raise ValueError(
            'Sequence stitching currently requires --batch_size 1. '
            'Increasing it does not preserve sequence-window semantics.'
        )
    if args.test_workers < 0:
        raise ValueError('test_workers must be non-negative.')
    if args.prefetch_factor <= 0:
        raise ValueError('prefetch_factor must be positive.')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = args.logpath + 'sem_seg/' + args.log_dir
    if args.visual:
        if args.visual_count > 0:
            visual_name = 'visual_random_%d_seed%d' % (
                args.visual_count,
                args.visual_seed,
            )
        else:
            visual_name = 'visual'
        visual_dir = Path(experiment_dir) / visual_name
        if args.visual_count > 0 and visual_dir.exists() and any(visual_dir.iterdir()):
            raise FileExistsError(
                'Random visualization directory is not empty: %s. '
                'Rename/remove it or choose another --visual_seed.' % visual_dir
            )
        visual_dir.mkdir(parents=True, exist_ok=True)
        visual_rng = np.random.default_rng(args.visual_seed)
        visual_reservoir = []
        visual_seen = 0
    if args.centroid_txt:
        centroid_dir = Path(args.centroid_dir) if args.centroid_dir else Path(experiment_dir) / 'out_centroid'
        centroid_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.epoch is None:
        file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    else:
        file_handler = logging.FileHandler('%s/eval_epoch-%d.txt' % (experiment_dir, args.epoch))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.datapath
    NUM_CLASSES = 1
    SEQ_LEN = args.seqlen
    BATCH_SIZE = args.batch_size


    print("start loading test data ...")
    TEST_DATASET = TestIRSeqDataLoader(
        args.dataset,
        data_root=root,
        seq_len=SEQ_LEN,
        cat_len=int(SEQ_LEN * 0.1),
        transform=None,
        load_annotations=(not args.output_only or args.attribution),
    )
    sequence_datasets = [
        TEST_DATASET[seq_index]
        for seq_index in range(len(TEST_DATASET))
    ]
    flattened_test_dataset = torch.utils.data.ConcatDataset(sequence_datasets)
    test_dataloader = torch.utils.data.DataLoader(
        flattened_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        **test_loader_options(args.test_workers, args.prefetch_factor)
    )

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    sys.path.append(experiment_dir)
    MODEL = importlib.import_module(model_name)
    # detector = torch.nn.DataParallel(MODEL.generator(NUM_CLASSES, SEQ_LEN)).cuda()
    detector = MODEL.detector(NUM_CLASSES, SEQ_LEN, SEQ_LEN)
    if args.epoch is None:
        checkpoint_path = str(experiment_dir) + '/checkpoints/best_model.pth'
    else:
        checkpoint_path = (
            str(experiment_dir)
            + '/checkpoints/epoch_%d_model.pth' % args.epoch
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # ## multi-GPU models load on single-GPU device
    # new_state_dict = OrderedDict()
    # for k,v in checkpoint['model_state_dict'].items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # detector.load_state_dict(new_state_dict)   ## or use the above detector definition
    # ## ##########################################
    detector.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    detector = detector.cuda().eval()
    evaluator = None if args.output_only else ShootingRules()

    with torch.inference_mode():
        num_batches = len(test_dataloader)
        total_intersection_mid = 0
        total_union_mid = 0
        total_predicted_positive_mid = 0
        total_target_positive_mid = 0

        Th_Seg = np.array([0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, .35, 0.4,
                           .45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1])
        FalseNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        TrueNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        TgtNumAll = np.zeros([len(TEST_DATASET),len(Th_Seg)])
        pixelsNumber = np.zeros(len(TEST_DATASET))


        # if args.attribution:
        #     path_interpolation_func = ZeroLinearPath(fold=50)

        log_string('---- EVALUATION----')

        time_start = time.time()
        test_iterator = iter(test_dataloader)
        for seq_idx, seq_dataset in tqdm(
            enumerate(sequence_datasets),
            total=len(sequence_datasets),
            smoothing=0.9,
        ):
            storage_length = sequence_storage_length(seq_dataset, SEQ_LEN)
            seq_midpred_all = None
            targets_all = None
            centroids_all = None
            sequence_spatial_size = None
            for i in range(len(seq_dataset)):
                images, targets, centroids, first_end = next(test_iterator)
                sequence_spatial_size = tuple(images.shape[-2:])
                images = images.float().cuda(non_blocking=True)
                first_frame, end_frame = first_end
                first_frame = int(first_frame.item())
                end_frame = int(end_frame.item())

                if args.attribution:
                    paths = [os.path.join(TEST_DATASET.seq_names[seq_idx], '%05d.png' % (fi+1))
                             for fi in range(first_frame, end_frame+1)]
                    savepath = os.path.join(experiment_dir, 'Attribution_ZeroLinearPath_0.1')
                    # seq_midpred = IR_Integrated_gradient(images, targets, (paths, args.dataset, savepath), detector, path_interpolation_func)

                else:
                    seq_features, seq_midpred = detector(images)
                    del seq_features
                    window_prediction = torch.sigmoid(seq_midpred).cpu()
                    del seq_midpred, images

                    if seq_midpred_all is None:
                        seq_midpred_all = torch.zeros(
                            (1, storage_length) + tuple(window_prediction.shape[-2:]),
                            dtype=window_prediction.dtype,
                        )
                        if not args.output_only:
                            targets_all = torch.zeros(
                                (1, storage_length) + tuple(targets.shape[-2:]),
                                dtype=targets.dtype,
                            )
                            centroids_all = torch.zeros(
                                (1, storage_length) + tuple(centroids.shape[-2:]),
                                dtype=centroids.dtype,
                            )

                    merge_prediction_window(
                        seq_midpred_all,
                        window_prediction,
                        first_frame,
                    )
                    if not args.output_only:
                        copy_annotation_window(
                            targets_all,
                            targets,
                            first_frame,
                        )
                        copy_annotation_window(
                            centroids_all,
                            centroids,
                            first_frame,
                        )
                    del window_prediction, targets, centroids

            if not args.attribution:
                seq_name = TEST_DATASET.seq_names[seq_idx]
                centroid_lines = []
                if not args.output_only:
                    ############### for IoU ###############
                    pred_choice_mid = seq_midpred_all.numpy() > args.threshold_eval
                    batch_label = targets_all.numpy() > 0
                    total_intersection_mid += np.count_nonzero(
                        np.logical_and(pred_choice_mid, batch_label)
                    )
                    total_union_mid += np.count_nonzero(
                        np.logical_or(pred_choice_mid, batch_label)
                    )
                    total_predicted_positive_mid += np.count_nonzero(
                        pred_choice_mid
                    )
                    total_target_positive_mid += np.count_nonzero(batch_label)

                ############### for Pd&Fa ###############
                _, t, h, w = seq_midpred_all.size()
                if not args.output_only:
                    pixelsNumber[seq_idx] += t * h * w
                for ti in range(t):
                    midpred_ti = seq_midpred_all[:, ti, :, :].numpy()
                    if not args.output_only:
                        centroid_ti = centroids_all[:, ti, :, :].numpy()
                        expected_h, expected_w = centroid_ti.shape[-2:]
                    else:
                        expected_h, expected_w = sequence_spatial_size
                    if midpred_ti.shape[-2:] != (expected_h, expected_w):
                        midpred_ti = cv2.resize(
                            midpred_ti[0, :, :],
                            (expected_w, expected_h),
                        )[None, :, :]
                    if not args.output_only:
                        false_numbers, true_numbers, target_numbers = (
                            evaluator.evaluate_thresholds(
                                midpred_ti,
                                centroid_ti,
                                Th_Seg,
                            )
                        )
                        FalseNumAll[seq_idx, :] += false_numbers
                        TrueNumAll[seq_idx, :] += true_numbers
                        TgtNumAll[seq_idx, :] += target_numbers

                    ############### save results ###############
                    plus1 = 0 if args.dataset == 'RGB-T' else 1
                    frame_idx = ti + plus1
                    png_name = '%05d.png' % frame_idx

                    if args.centroid_txt:
                        binary_mask = np.uint8(
                            midpred_ti.squeeze(0) > args.centroid_threshold
                        ) * 255
                        frame_centroids = calculate_centroids(binary_mask)
                        centroid_lines.append(
                            format_centroid_line(frame_idx, frame_centroids)
                        )

                    if args.visual:
                        visual_array = np.uint8(midpred_ti.squeeze(0) * 255)
                        visual_item = (seq_name, png_name, visual_array)
                        visual_seen += 1
                        if args.visual_count == 0:
                            seq_dir = visual_dir / seq_name
                            seq_dir.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(visual_array).save(seq_dir / png_name)
                        elif len(visual_reservoir) < args.visual_count:
                            visual_reservoir.append(visual_item)
                        else:
                            replace_idx = int(visual_rng.integers(0, visual_seen))
                            if replace_idx < args.visual_count:
                                visual_reservoir[replace_idx] = visual_item
                        # scio.savemat(os.path.join(seq_dir, '%05d.mat' % frame_idx), {'TestOut': midpred_ti.squeeze(0)})

                if args.centroid_txt:
                    output_txt = centroid_dir / ('%s.txt' % seq_name)
                    with output_txt.open('w') as output_file:
                        output_file.write('\n'.join(centroid_lines))
                        output_file.write('\n')
                del seq_midpred_all
                if not args.output_only:
                    del targets_all, centroids_all

        if args.visual and args.visual_count > 0:
            for seq_name, png_name, visual_array in visual_reservoir:
                seq_dir = visual_dir / seq_name
                seq_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(visual_array).save(seq_dir / png_name)
            log_string(
                'Saved %d randomly sampled visualizations from %d frames.'
                % (len(visual_reservoir), visual_seen)
            )
        if args.centroid_txt:
            log_string('Centroid TXT files saved to %s.' % centroid_dir)

        time_end = time.time()
        log_string(
            'Evaluation elapsed time: %.2f seconds for %d windows.'
            % (time_end - time_start, num_batches)
        )
        # print('FPS=%.3f' % (2000*1.2 / (time_end - time_start)))
        ############### log Pd&Fa results ###############
        if not args.attribution and not args.output_only:
            if 'NUDT-MIRSDT' in args.dataset:
                writeNUDTMIRSDT_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid,
                                    total_union_mid, Th_Seg, TEST_DATASET, log_string)
            else:
                writeMIRST_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid,
                               total_union_mid, Th_Seg, TEST_DATASET, log_string)
            pixel_precision = total_intersection_mid / max(
                total_predicted_positive_mid, 1
            )
            pixel_recall = total_intersection_mid / max(
                total_target_positive_mid, 1
            )
            pixel_f1 = 2 * total_intersection_mid / max(
                total_predicted_positive_mid + total_target_positive_mid,
                1,
            )
            log_string('Eval pixel precision: %f' % pixel_precision)
            log_string('Eval pixel recall: %f' % pixel_recall)
            log_string('Eval pixel F1: %f' % pixel_f1)

        if args.profile_flops:
            flops, params = profile(detector, inputs=(torch.randn(1, 1, args.seqlen, 200, 300).cuda(),))
            flops, params = clever_format([flops, params], '%.3f')
            print('FLOPS for %d frames: ' % SEQ_LEN, flops)
            print('Params:', count_parameters(detector))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
