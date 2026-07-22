"""
Author: Benny
Date: Nov 2019
"""
import argparse
import inspect
import os
from data_utils.TestDataLoader import TestIRSeqDataLoader
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import time
from PIL import Image
import cv2
import torch.nn.functional as F
from ShootingRules import ShootingRules
from sequence_utils import SequenceAccumulator, frame_range_length
from runtime_utils import load_checkpoint, parse_visible_devices
# from attribution.core import IR_Integrated_gradient, MeanLinearPath, ZeroLinearPath
from write_results import writeNUDTMIRSDT_ROC, writeMIRST_ROC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'networks/models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch of generator to test [default: None]')
    parser.add_argument(
        '--gpu', type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        help='visible GPU device(s); defaults to the existing CUDA_VISIBLE_DEVICES value',
    )
    parser.add_argument('--seqlen', type=int, default=40, help='Frame number as an input [default: 100]')
    parser.add_argument('--datapath', type=str, default='./datasets/NUDT-MIRSDT', help='Data path')
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT', help='dataset name [default: NUDT-MIRSDT, IRDST-simulation, RGB-T, SatVideoIRSDT, SatVideoIRSDT_v1]')
    parser.add_argument('--log_dir', type=str, default='NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40', help='experiment root')
    parser.add_argument('--logpath', type=str, default='./log/', help='Log path: ./log/')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--threshold_eval', type=float, default=0.5, help='Threshold in evaluation [default: 0.5]')
    parser.add_argument('--attribution', action='store_true', default=False, help='This test is attribution analysis or not')
    parser.add_argument('--profile_model', type=int, default=0, choices=[0, 1],
                        help='Run optional THOP profiling after evaluation [default: 0]')
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1],
                        help='Use CUDA automatic mixed precision [default: 0]')
    parser.add_argument('--eval_chunk_rows', type=int, default=None,
                        help='Override checkpoint eval row chunking; 0 disables it')
    parser.add_argument('--overwrite_outputs', action='store_true', default=False)
    return parser.parse_args()


def count_parameters(model):
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1000 ** 2)  # MB (十进制)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    visible_devices = parse_visible_devices(args.gpu)
    if len(visible_devices) != 1:
        raise ValueError('test_BRTD.py requires exactly one GPU.')
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[0]
    if args.batch_size != 1:
        raise ValueError('test_BRTD.py only supports --batch_size 1.')
    if args.attribution:
        raise NotImplementedError(
            '--attribution is unavailable because its implementation is commented out.'
        )

    experiment_root = Path(args.logpath).expanduser().resolve() / 'sem_seg'
    experiment_dir = (experiment_root / args.log_dir).resolve()
    try:
        experiment_dir.relative_to(experiment_root)
    except ValueError as error:
        raise ValueError('--log_dir must be under %s.' % experiment_root) from error
    if not experiment_dir.is_dir():
        raise FileNotFoundError(experiment_dir)
    if args.visual:
        visual_dir = experiment_dir / 'visual'
        if (
            visual_dir.exists()
            and any(visual_dir.iterdir())
            and not args.overwrite_outputs
        ):
            raise FileExistsError(
                'Visual directory is not empty: %s. Pass --overwrite_outputs '
                'only when replacement is intentional.' % visual_dir
            )
        visual_dir.mkdir(exist_ok=True, parents=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.epoch is None:
        file_handler = logging.FileHandler(experiment_dir / 'eval.txt')
    else:
        file_handler = logging.FileHandler(experiment_dir / ('eval_epoch-%d.txt' % args.epoch))
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
    TEST_DATASET  = TestIRSeqDataLoader(args.dataset, data_root=root,  seq_len=SEQ_LEN, cat_len=int(SEQ_LEN*0.1), transform=None)

    '''MODEL LOADING'''
    if args.epoch is None:
        checkpoint_path = experiment_dir / 'checkpoints' / 'best_model.pth'
    else:
        checkpoint_path = experiment_dir / 'checkpoints' / ('epoch_%d_model.pth' % args.epoch)
    checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
    model_name = checkpoint.get('model_name')
    if model_name is None:
        log_files = sorted((experiment_dir / 'logs').glob('*.txt'))
        if len(log_files) != 1:
            raise RuntimeError(
                'Checkpoint has no model_name and exactly one model log could not be identified.'
            )
        model_name = log_files[0].stem
    model_path = (experiment_dir / ('%s.py' % model_name)).resolve()
    if model_path.parent != experiment_dir or not model_path.is_file():
        raise ValueError('Unsafe or missing model snapshot: %s' % model_name)
    sys.path.insert(0, str(experiment_dir))
    MODEL = importlib.import_module(model_name)
    constructor_parameters = inspect.signature(MODEL.detector).parameters
    model_config = {
        key: value for key, value in checkpoint.get('model_config', {}).items()
        if key in constructor_parameters
        and key not in {'spatial_ckpt', 'st_ckpt'}
    }
    if args.eval_chunk_rows is not None:
        model_config = dict(model_config)
        if 'eval_chunk_rows' not in constructor_parameters:
            if args.eval_chunk_rows:
                raise ValueError('%s does not support eval chunking.' % model_name)
        else:
            model_config['eval_chunk_rows'] = args.eval_chunk_rows
    detector = MODEL.detector(
        NUM_CLASSES,
        SEQ_LEN,
        SEQ_LEN,
        **model_config,
    ).cuda()
    # ## multi-GPU models load on single-GPU device
    # new_state_dict = OrderedDict()
    # for k,v in checkpoint['model_state_dict'].items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # detector.load_state_dict(new_state_dict)   ## or use the above detector definition
    # ## ##########################################
    state_dict = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in checkpoint['model_state_dict'].items()
    }
    detector.load_state_dict(state_dict)
    del checkpoint, state_dict
    detector.eval()
    evaluator = ShootingRules()

    with torch.inference_mode():
        num_batches = 0
        metric_counts = torch.zeros(2, device='cuda', dtype=torch.int64)

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
        if args.epoch is None:
            eval_desc = 'Test best_model'
        else:
            eval_desc = 'Test epoch_%03d' % args.epoch

        eval_bar = tqdm(
            enumerate(TEST_DATASET),
            total=len(TEST_DATASET),
            desc=eval_desc,
            smoothing=0.9,
            ascii=True,
            dynamic_ncols=False,
            ncols=100,
            mininterval=0.5,
            leave=True,
            file=sys.stdout,
            bar_format=(
                '{desc}: {percentage:3.0f}%|{bar:30}| '
                '{n_fmt}/{total_fmt} '
                '[{elapsed}<{remaining}, {rate_fmt}]'
            ),
        )

        for seq_idx, seq_dataset in eval_bar:
        # for seq_idx, seq_dataset in tqdm(enumerate(TEST_DATASET), total=len(TEST_DATASET), smoothing=0.9):
            seq_dataloader = torch.utils.data.DataLoader(seq_dataset, batch_size=BATCH_SIZE, shuffle=False)
            num_batches += len(seq_dataloader)
            accumulator = SequenceAccumulator()
            for i, (images, targets, centroids, first_end) in enumerate(seq_dataloader):
                images = images.float().cuda(non_blocking=True)
                targets = targets.float().cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                    sequence_features, seq_midpred = detector(images)
                    del sequence_features
                if seq_midpred.shape[-2:] != targets.shape[-2:]:
                    seq_midpred = F.interpolate(
                        seq_midpred,
                        size=targets.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                valid_length = frame_range_length(first_end)
                accumulator.add(
                    torch.sigmoid(seq_midpred[:, :valid_length]),
                    targets[:, :valid_length],
                    first_end,
                    centroids[:, :valid_length],
                )
                del images, targets, seq_midpred, centroids

            if accumulator.predictions is not None:
                seq_midpred_all = accumulator.predictions
                targets_all = accumulator.targets
                centroids_all = accumulator.centroids
                ############### for IoU ###############
                pred_choice_mid = seq_midpred_all.gt(args.threshold_eval)
                batch_label = targets_all.gt(0)
                metric_counts[0] += torch.logical_and(
                    pred_choice_mid, batch_label
                ).sum(dtype=torch.int64)
                metric_counts[1] += torch.logical_or(
                    pred_choice_mid, batch_label
                ).sum(dtype=torch.int64)
                seq_midpred_cpu = seq_midpred_all.cpu().numpy()
                centroids_cpu = centroids_all.numpy()
                del seq_midpred_all, targets_all, centroids_all
                del accumulator, pred_choice_mid, batch_label

                ############### for Pd&Fa ###############
                _, t, h, w = seq_midpred_cpu.shape
                pixelsNumber[seq_idx] += t * h * w
                for ti in range(t):
                    midpred_ti = seq_midpred_cpu[:, ti, :, :]
                    centroid_ti = centroids_cpu[:, ti, :, :]
                    if midpred_ti.shape[-2:] != centroid_ti.shape[-2:]:
                        h, w = centroid_ti.shape[-2:]
                        midpred_ti = cv2.resize(midpred_ti[0, :, :], (w, h))[None, :, :]
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
                    if args.visual:
                        midpred_ti_png = Image.fromarray(
                            np.uint8(midpred_ti.squeeze(0) * 255)
                        )
                        plus1 = 0 if args.dataset == 'RGB-T' else 1
                        png_name = '%05d.png' % (ti+1*plus1)
                        seq_dir = Path(os.path.join(visual_dir, TEST_DATASET.seq_names[seq_idx]))
                        seq_dir.mkdir(exist_ok=True)
                        midpred_ti_png.save(os.path.join(seq_dir, png_name))
                        # scio.savemat(os.path.join(seq_dir, '%05d.mat' % (ti+1*plus1)), {'TestOut': midpred_ti.squeeze(0)})
                del seq_midpred_cpu, centroids_cpu

        time_end = time.time()
        log_string(
            'Evaluation elapsed time: %.2f seconds for %d windows.'
            % (time_end - time_start, num_batches)
        )
        # print('FPS=%.3f' % (2000*1.2 / (time_end - time_start)))
        ############### log Pd&Fa results ###############
        total_intersection_mid, total_union_mid = metric_counts.tolist()
        if not args.attribution:
            if 'NUDT-MIRSDT' in args.dataset:
                writeNUDTMIRSDT_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid,
                                    total_union_mid, Th_Seg, TEST_DATASET, log_string)
            else:
                writeMIRST_ROC(FalseNumAll, TrueNumAll, TgtNumAll, pixelsNumber, total_intersection_mid,
                               total_union_mid, Th_Seg, TEST_DATASET, log_string)

        if args.profile_model:
            from thop import clever_format, profile

            profile_input = torch.randn(
                1, 1, args.seqlen, 200, 300,
                device=next(detector.parameters()).device,
            )
            flops, params = profile(detector, inputs=(profile_input,))
            flops, params = clever_format([flops, params], '%.3f')
            print('FLOPS for %d frames: ' % SEQ_LEN, flops)
        print('Params:', count_parameters(detector))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
