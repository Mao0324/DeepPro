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
import torch.nn.functional as F
from ShootingRules_v2 import ShootingRules
from sklearn.metrics import auc
from collections import OrderedDict
from sequence_utils import SequenceAccumulator
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
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT', help='dataset name [default: NUDT-MIRSDT, IRDST-simulation, RGB-T, SatVideoIRSDT]')
    parser.add_argument('--log_dir', type=str, default='NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40', help='experiment root')
    parser.add_argument('--logpath', type=str, default='./log/', help='Log path: ./log/')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--threshold_eval', type=float, default=0.5, help='Threshold in evaluation [default: 0.5]')
    parser.add_argument('--attribution', action='store_true', default=False, help='This test is attribution analysis or not')
    parser.add_argument('--profile_model', type=int, default=0, choices=[0, 1],
                        help='Run optional THOP profiling after evaluation [default: 0]')
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1],
                        help='Use CUDA automatic mixed precision [default: 0]')
    return parser.parse_args()


def count_parameters(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1000 ** 2)  # MB (十进制)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.batch_size != 1:
        raise ValueError('test_BRTD.py only supports --batch_size 1.')
    if args.attribution:
        raise NotImplementedError(
            '--attribution is unavailable because its implementation is commented out.'
        )

    experiment_dir = Path(args.logpath) / 'sem_seg' / args.log_dir
    if args.visual:
        visual_dir = experiment_dir / 'visual'
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_name = checkpoint.get('model_name')
    if model_name is None:
        log_files = sorted((experiment_dir / 'logs').glob('*.txt'))
        if len(log_files) != 1:
            raise RuntimeError(
                'Checkpoint has no model_name and exactly one model log could not be identified.'
            )
        model_name = log_files[0].stem
    sys.path.append(str(experiment_dir))
    MODEL = importlib.import_module(model_name)
    model_config = checkpoint.get('model_config', {})
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
    detector.eval()
    eval = ShootingRules()

    with torch.no_grad():
        num_batches = 0
        total_intersection_mid = 0
        total_union_mid = 0

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
                images, targets = images.float().cuda(), targets.float().cuda()
                first_frame, end_frame = first_end

                if args.attribution:
                    paths = [os.path.join(TEST_DATASET.seq_names[seq_idx], '%05d.png' % (fi+1))
                             for fi in range(first_frame, end_frame+1)]
                    savepath = os.path.join(experiment_dir, 'Attribution_ZeroLinearPath_0.1')
                    # seq_midpred = IR_Integrated_gradient(images, targets, (paths, args.dataset, savepath), detector, path_interpolation_func)

                else:
                    with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                        _, seq_midpred = detector(images)   ## b, t, h, w
                    if seq_midpred.shape[-2:] != targets.shape[-2:]:
                        seq_midpred = F.interpolate(seq_midpred, size=targets.shape[-2:])
                    accumulator.add(
                        torch.sigmoid(seq_midpred).cpu(),
                        targets.cpu(),
                        first_end,
                        centroids,
                    )

            if not args.attribution:
                seq_midpred_all = accumulator.predictions
                targets_all = accumulator.targets
                centroids_all = accumulator.centroids
                ############### for IoU ###############
                pred_choice_mid = (seq_midpred_all.numpy() > args.threshold_eval) * 1.
                batch_label     = targets_all.numpy()
                total_intersection_mid += np.sum(pred_choice_mid * batch_label)
                total_union_mid += ((pred_choice_mid + batch_label) > 0).astype(np.float32).sum()

                ############### for Pd&Fa ###############
                _, t, h, w = seq_midpred_all.size()
                pixelsNumber[seq_idx] += t * h * w
                for ti in range(t):
                    midpred_ti = seq_midpred_all[:, ti, :, :].numpy().copy()
                    centroid_ti  = centroids_all[:, ti, :, :].numpy().copy()
                    if midpred_ti.shape[-2:] != centroid_ti.shape[-2:]:
                        h, w = centroid_ti.shape[-2:]
                        midpred_ti = cv2.resize(midpred_ti[0, :, :], (w, h))[None, :, :]
                    for th_i in range(len(Th_Seg)):
                        FalseNum, TrueNum, TgtNum = eval(midpred_ti, centroid_ti, Th_Seg[th_i])
                        FalseNumAll[seq_idx, th_i] = FalseNumAll[seq_idx, th_i] + FalseNum
                        TrueNumAll[seq_idx, th_i]  = TrueNumAll[seq_idx, th_i] + TrueNum
                        TgtNumAll[seq_idx, th_i]   = TgtNumAll[seq_idx, th_i] + TgtNum

                    ############### save results ###############
                    if args.visual:
                        midpred_ti_png = Image.fromarray(uint8(midpred_ti.squeeze(0) * 255))
                        plus1 = 0 if args.dataset == 'RGB-T' else 1
                        png_name = '%05d.png' % (ti+1*plus1)
                        seq_dir = Path(os.path.join(visual_dir, TEST_DATASET.seq_names[seq_idx]))
                        seq_dir.mkdir(exist_ok=True)
                        midpred_ti_png.save(os.path.join(seq_dir, png_name))
                        # scio.savemat(os.path.join(seq_dir, '%05d.mat' % (ti+1*plus1)), {'TestOut': midpred_ti.squeeze(0)})

        time_end = time.time()
        # print('FPS=%.3f' % (2000*1.2 / (time_end - time_start)))
        ############### log Pd&Fa results ###############
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
