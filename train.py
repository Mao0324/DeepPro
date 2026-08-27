"""
Author: Benny
Date: Nov 2019
"""
import argparse
from contextlib import nullcontext
import gc
import inspect
import os
from data_utils.TrainDataLoader import (
    SequenceGeometryAugmentation,
    TrainIRSeqDataLoader,
)
from data_utils.TestDataLoader import TestIRSeqDataLoader
from networks.losses import (
    LOSS_DESCRIPTIONS,
    LOSS_NAMES,
    build_segmentation_loss,
    loss_experiment_name,
)
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
import subprocess
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from runtime_utils import (
    all_reduce_sum,
    atomic_torch_save,
    broadcast_object,
    distributed_barrier,
    finalize_distributed,
    initialize_distributed,
    launch_with_torchrun_if_needed,
    load_checkpoint,
    move_optimizer_state,
    parse_visible_devices,
    unwrap_model,
)
from sequence_utils import SequenceAccumulator, frame_range_length

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'networks/models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def seed_everything(seed=46, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def model_configuration(model_module, args):
    """Build supported constructor options and persist them in checkpoints."""
    parameters = inspect.signature(model_module.detector).parameters
    configuration = {}
    if 'eval_chunk_rows' in parameters:
        configuration['eval_chunk_rows'] = args.eval_chunk_rows
    elif args.eval_chunk_rows:
        raise ValueError(
            '%s does not support --eval_chunk_rows.' % args.model
        )
    if 'spatial_ckpt' in parameters:
        configuration.update({
            'spatial_ckpt': args.spatial_ckpt,
            'st_ckpt': args.st_ckpt,
            'freeze_pretrained': bool(args.freeze_pretrained),
        })
    brtd_options = {
        'use_background': bool(args.brtd_use_background),
        'adaptive_tdc': bool(args.brtd_adaptive_tdc),
        'use_gate': bool(args.brtd_use_gate),
        'zero_init': bool(args.brtd_zero_init),
    }
    for name, value in brtd_options.items():
        if name in parameters:
            configuration[name] = value
    structure_options = {
        'structure_variant': args.structure_variant,
        'structure_bottleneck_channels': (
            args.structure_bottleneck_channels
        ),
        'structure_max_shift': args.structure_max_shift,
    }
    for name, value in structure_options.items():
        if name in parameters:
            configuration[name] = value
    feedback_options = {
        'feedback_interval': args.feedback_interval,
        'feedback_alignment_levels': args.feedback_alignment_levels,
        'feedback_eval_tile_size': args.feedback_eval_tile_size,
        'feedback_eval_tile_overlap': args.feedback_eval_tile_overlap,
    }
    for name, value in feedback_options.items():
        if name in parameters:
            configuration[name] = value
    return configuration


def clean_model_state_dict(state_dict):
    """Remove a DDP prefix without changing ordinary checkpoint keys."""
    if state_dict and all(key.startswith('module.') for key in state_dict):
        return {
            key[len('module.'):]: value for key, value in state_dict.items()
        }
    return state_dict


def make_checkpoint_state(
    detector,
    optimizer,
    grad_scaler,
    epoch,
    best_iou,
    args,
    config,
    early_stopping_state=None,
):
    stored_config = dict(config)
    if 'spatial_ckpt' in stored_config:
        # Branch weights are already part of model_state_dict. Test/resume must
        # not depend on, or unexpectedly reopen, the original pretrain paths.
        stored_config['spatial_ckpt'] = None
        stored_config['st_ckpt'] = None
        stored_config['freeze_pretrained'] = False
    state = {
        'epoch': epoch,
        'class_avg_iou': best_iou,
        'model_name': args.model,
        'model_config': stored_config,
        'model_state_dict': unwrap_model(detector).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'grad_scaler_state_dict': grad_scaler.state_dict(),
    }
    if early_stopping_state is not None:
        state['early_stopping_state'] = dict(early_stopping_state)
    return state


def new_early_stopping_state(metric):
    """Return serializable state for validation-based early stopping."""
    mode = 'min' if metric == 'eval_loss' else 'max'
    return {
        'metric': metric,
        'mode': mode,
        'best_value': None,
        'best_epoch': 0,
        'bad_epochs': 0,
        'stopped': False,
    }


def early_stopping_metric_value(metric, eval_loss, eval_iou, eval_f1):
    values = {
        'eval_loss': eval_loss,
        'eval_iou': eval_iou,
        'eval_f1': eval_f1,
    }
    return float(values[metric])


def update_early_stopping_state(
    state,
    value,
    epoch,
    patience,
    min_delta,
    start_epoch,
):
    """Update early-stopping state and return (improved, should_stop)."""
    best_value = state['best_value']
    if best_value is None:
        improved = True
    elif state['mode'] == 'min':
        improved = value < best_value - min_delta
    else:
        improved = value > best_value + min_delta

    if improved:
        state['best_value'] = float(value)
        state['best_epoch'] = int(epoch)
        state['bad_epochs'] = 0
    elif epoch >= start_epoch:
        state['bad_epochs'] = int(state['bad_epochs']) + 1
    else:
        state['bad_epochs'] = 0

    should_stop = (
        epoch >= start_epoch and state['bad_epochs'] >= patience
    )
    state['stopped'] = bool(should_stop)
    return improved, should_stop


def evaluate_sequences(
    detector,
    criterion,
    sequence_datasets,
    validation_loader,
    device,
    threshold,
    epoch,
    show_progress,
    use_amp=False,
):
    """Return local validation sums after overlap-aware stitching."""
    detector.eval()
    metric_counts = torch.zeros(3, dtype=torch.int64)
    loss_sum = 0.0
    loss_count = 0
    validation_iterator = iter(validation_loader)
    previous_spatial_size = None

    with torch.inference_mode():
        for sequence_dataset in tqdm(
            sequence_datasets,
            total=len(sequence_datasets),
            smoothing=0.9,
            disable=not show_progress,
        ):
            accumulator = SequenceAccumulator()
            for _ in range(len(sequence_dataset)):
                images, targets, _centroids, first_end = next(
                    validation_iterator
                )
                spatial_size = tuple(images.shape[-2:])
                if (
                    previous_spatial_size is not None
                    and spatial_size != previous_spatial_size
                ):
                    torch.cuda.empty_cache()
                previous_spatial_size = spatial_size
                images = images.float().to(device, non_blocking=True)
                targets = targets.float().to(device, non_blocking=True)
                with torch.autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    sequence_features, sequence_logits = detector(images)
                del sequence_features
                if sequence_logits.shape[-2:] != targets.shape[-2:]:
                    sequence_logits = F.interpolate(
                        sequence_logits,
                        size=targets.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                valid_length = frame_range_length(first_end)
                valid_logits = sequence_logits[:, :valid_length]
                valid_targets = targets[:, :valid_length]
                valid_images = images[:, :, :valid_length]
                with torch.autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    window_loss = criterion(
                        valid_logits,
                        valid_targets,
                        images=valid_images,
                        epoch=epoch,
                    )
                loss_sum += float(window_loss.detach().cpu())
                loss_count += 1
                accumulator.add(
                    torch.sigmoid(valid_logits.float()).cpu(),
                    valid_targets.cpu(),
                    first_end,
                )
                del window_loss
                del sequence_logits, valid_logits, valid_targets, valid_images
                del images, targets, _centroids

            predicted = accumulator.predictions.gt(threshold)
            target = accumulator.targets.gt(0)
            metric_counts[0] += torch.logical_and(predicted, target).sum(
                dtype=torch.int64
            )
            metric_counts[1] += predicted.sum(dtype=torch.int64)
            metric_counts[2] += target.sum(dtype=torch.int64)
            del accumulator, predicted, target

    if loss_count == 0:
        raise RuntimeError('Validation loader contains no windows.')
    return loss_sum, loss_count, metric_counts


def multiprocessing_loader_options(worker_count, prefetch_factor):
    """Return DataLoader options that are valid with and without workers."""
    options = {'num_workers': worker_count}
    if worker_count > 0:
        options.update({
            'persistent_workers': True,
            'prefetch_factor': prefetch_factor,
        })
    return options


def binary_segmentation_metrics(
    true_positive,
    predicted_positive,
    target_positive,
):
    """Return micro-averaged pixel IoU, precision, recall and F1."""
    true_positive = true_positive.to(torch.float64)
    predicted_positive = predicted_positive.to(torch.float64)
    target_positive = target_positive.to(torch.float64)
    union = predicted_positive + target_positive - true_positive
    metrics = torch.stack((
        true_positive / union.clamp_min(1),
        true_positive / predicted_positive.clamp_min(1),
        true_positive / target_positive.clamp_min(1),
        2 * true_positive / (predicted_positive + target_positive).clamp_min(1),
    ))
    return tuple(metrics.tolist())


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='DeepPro-Plus', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Accumulate this many physical batches before each optimizer step.',
    )
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.005, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--gpu_num', type=int, default=1, help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--datapath', type=str, default='./datasets/NUDT-MIRSDT')
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT', help='dataset name [default: NUDT-MIRSDT, NUDT-MIRSDT-HiNo, '
                                            'RGB-T, SatVideoIRSDT, IRDST-simulation, IRSatVideo-LEO]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--savepath', type=str, default='./log/', help='Save path [default: ./log/]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--seqlen', type=int, default=40, help='Frame number as an input [default: 100]')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch Size for train generator [default: 128, 72]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--sample_rate', type=float, default=0.1, help='Sampling rate for training [default: 0.1(NUDT-MIRSDT), '
                                                                     '0.03(IRDST), 0.05(RGB-T), 0.04(SatVideoIRSDT)]')
    parser.add_argument(
        '--sequence_augmentation', type=int, default=0, choices=[0, 1],
        help='Enable spatial symmetry and temporal reversal augmentation.',
    )
    parser.add_argument(
        '--mask_padded_frames', type=int, default=0, choices=[0, 1],
        help='Exclude synthetic all-zero sequence-padding frames '
             'from training loss and metrics (F1-OHEM only).',
    )
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--threshold_eval', type=float, default=0.5, help='Threshold in evaluation [default: 0.5]')
    parser.add_argument(
        '--early_stopping_patience', type=int, default=0,
        help='Validation epochs without improvement before stopping; '
             '0 disables early stopping [default: 0].',
    )
    parser.add_argument(
        '--early_stopping_min_delta', type=float, default=1.0e-4,
        help='Minimum monitored-metric improvement [default: 1e-4].',
    )
    parser.add_argument(
        '--early_stopping_start_epoch', type=int, default=15,
        help='First epoch at which non-improving evaluations count toward '
             'patience [default: 15].',
    )
    parser.add_argument(
        '--early_stopping_metric',
        choices=['eval_f1', 'eval_iou', 'eval_loss'],
        default='eval_f1',
        help='Validation metric monitored by early stopping '
             '[default: eval_f1].',
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='soft_iou',
        choices=LOSS_NAMES,
        help='Training loss. Default soft_iou exactly preserves the old behavior.soft_iou，' \
        'frame_soft_iou，bce，focal，dice，bce_dice，tversky，focal_tversky，lovasz，sls_iou，tda_sls,hard_focal,tversky_hard_focal,stc_f1',
    )
    parser.add_argument('--loss_eps', type=float, default=1.0,
                        help='Smoothing for Dice/Tversky/frame SoftIoU [default: 1.0]')
    parser.add_argument('--sls_eps', type=float, default=1e-6,
                        help='Numerical epsilon for SLS/TDA [default: 1e-6]')
    parser.add_argument('--focal_alpha', type=float, default=0.75,
                        help='Positive-class weight for Focal losses [default: 0.75]')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focusing exponent for Focal losses [default: 2.0]')
    parser.add_argument('--tversky_fp_weight', type=float, default=0.6,
                        help='False-positive weight in Tversky [default: 0.6]')
    parser.add_argument('--tversky_fn_weight', type=float, default=0.4,
                        help='False-negative weight in Tversky [default: 0.4]')
    parser.add_argument('--tversky_gamma', type=float, default=1.33,
                        help='Focal-Tversky exponent [default: 1.33]')
    parser.add_argument('--bce_weight', type=float, default=0.5,
                        help='BCE fraction in bce_dice [default: 0.5]')
    parser.add_argument('--hard_negative_topk', type=int, default=4096,
                        help='Hard background pixels retained per video clip [default: 4096]')
    parser.add_argument('--hard_focal_weight', type=float, default=0.25,
                        help='Hard-Focal coefficient in combined losses [default: 0.25]')
    parser.add_argument('--f1_ohem_dice_weight', type=float, default=0.15)
    parser.add_argument('--f1_ohem_hard_weight', type=float, default=0.10)
    parser.add_argument('--f1_ohem_negative_ratio', type=float, default=4.0)
    parser.add_argument('--f1_ohem_min_negatives', type=int, default=256)
    parser.add_argument('--f1_ohem_margin', type=float, default=1.0)
    parser.add_argument('--f1_ohem_warmup_epochs', type=int, default=5)
    parser.add_argument('--f1_ohem_ramp_epochs', type=int, default=10)
    parser.add_argument('--sls_location_weight', type=float, default=1.0,
                        help='Location coefficient in SLS/TDA [default: 1.0]')
    parser.add_argument('--sls_warmup_epochs', type=int, default=5,
                        help='Epochs before enabling SLS location term [default: 5]')
    parser.add_argument('--tda_weight', type=float, default=0.2,
                        help='Local TDA coefficient in tda_sls [default: 0.2]')
    parser.add_argument('--tda_mean_size', type=float, default=0.0,
                        help='Dataset mean target area; 0 uses current batch [default: 0]')
    parser.add_argument('--tda_mean_contrast', type=float, default=0.0,
                        help='Dataset mean local contrast; 0 uses current batch [default: 0]')
    parser.add_argument('--tda_dilation', type=int, default=3,
                        help='TDA object-box dilation in pixels [default: 3]')
    parser.add_argument('--stc_center_weight', type=float, default=0.1,
                        help='Center-response coefficient in stc_f1 [default: 0.1]')
    parser.add_argument('--stc_temporal_weight', type=float, default=0.05,
                        help='Temporal-consistency coefficient in stc_f1 [default: 0.05]')
    parser.add_argument('--stc_warmup_epochs', type=int, default=5,
                        help='Epochs before enabling STC auxiliary terms [default: 5]')
    parser.add_argument('--train_workers', type=int, default=8,
                        help='Persistent DataLoader workers used for training [default: 8]')
    parser.add_argument('--val_workers', type=int, default=4,
                        help='Persistent DataLoader workers used for validation [default: 4]')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Batches prefetched by each DataLoader worker [default: 2]')
    parser.add_argument('--use_swanlab', type=int, default=1, choices=[0, 1], help='Use SwanLab logging [default: 1]')
    parser.add_argument('--swanlab_project', type=str, default='DeepPro', help='SwanLab project name')
    parser.add_argument('--swanlab_workspace', type=str, default=None,
                        help='Optional SwanLab workspace/organization')
    parser.add_argument('--swanlab_group', type=str, default=None,
                        help='Optional SwanLab comparison group')
    parser.add_argument('--swanlab_mode', choices=['cloud', 'local', 'offline'],
                        default='cloud', help='SwanLab logging mode [default: cloud]')
    parser.add_argument('--swanlab_id', type=str, default=None,
                        help='Existing SwanLab run ID used for continuation')
    parser.add_argument('--swanlab_resume', choices=['allow', 'must', 'never'],
                        default='never', help='SwanLab run resume policy')
    parser.add_argument("--spatial_ckpt", type=str, default="")
    parser.add_argument("--st_ckpt", type=str, default="")
    parser.add_argument("--freeze_pretrained", type=int, default=0)
    parser.add_argument('--eval_chunk_rows', type=int, default=0,
                        help='TPro evaluation row chunk size; 0 disables chunking')
    parser.add_argument(
        '--train_amp', type=int, default=0, choices=[0, 1],
        help='Use FP16 CUDA autocast with FP32 loss and GradScaler during training.',
    )
    parser.add_argument(
        '--eval_amp', type=int, default=0, choices=[0, 1],
        help='Use FP16 CUDA autocast during validation [default: 0].',
    )
    parser.add_argument(
        '--eval_interval', type=int, default=1,
        help='Run full validation every N epochs and always on the final epoch.',
    )
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--deterministic', type=int, default=0, choices=[0, 1],
                        help='Use deterministic cuDNN kernels (may reduce speed)')
    parser.add_argument('--resume', choices=['auto', 'never'], default='auto',
                        help='Resume a valid checkpoint or refuse unsafe overwrite')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Explicit checkpoint to resume')
    parser.add_argument('--run_test_after_train', type=int, default=1,
                        choices=[0, 1], help='Run test.py after successful training')
    parser.add_argument('--base_ckpt', type=str, default='',
                        help='Backbone checkpoint used to initialize an adapter model')
    parser.add_argument('--base_lr_mult', type=float, default=1.0,
                        help='Learning-rate multiplier for non-BRTD parameters [default: 1.0]')
    parser.add_argument('--brtd_use_background', type=int, default=1,
                        choices=[0, 1])
    parser.add_argument('--brtd_adaptive_tdc', type=int, default=1,
                        choices=[0, 1])
    parser.add_argument('--brtd_use_gate', type=int, default=1,
                        choices=[0, 1])
    parser.add_argument('--brtd_zero_init', type=int, default=1,
                        choices=[0, 1])
    parser.add_argument(
        '--structure_variant', type=str, default='second_order',
        choices=[
            'raw_apmd', 'raw_apmd_rms', 'raw_apmd_channel_rms',
            'raw_apmd_motion_detrend',
            'raw_apmd_multiscale_contrast',
            'raw_apmd_hybrid_rms',
            'raw_apmd_hybrid_rms_scratch_init',
            'raw_apmd_hybrid_rms_scratch_bandpass',
            'raw_apmd_hybrid_rms_scratch_detail',
            'raw_apmd_hybrid_rms_motion_detrend',
            'raw_apmd_hybrid_rms_multiscale_contrast',
            'raw_apmd_hybrid_rms_motion_detrend_multiscale_contrast',
            'second_order',
            'lfp_shallow', 'lfp_deep',
            'global_align', 'local_align', 'multiscale_head',
            'bidirectional', 'tdc_dual_stream',
        ],
        help='Structural adapter used by DeepPro-Plus_BRTD3.',
    )
    parser.add_argument(
        '--structure_bottleneck_channels', type=int, default=8,
        help='Bottleneck width for BRTD3 structural adapters.',
    )
    parser.add_argument(
        '--structure_max_shift', type=float, default=4.0,
        help='Maximum alignment displacement in feature pixels.',
    )
    parser.add_argument('--feedback_interval', type=int, default=2)
    parser.add_argument('--feedback_alignment_levels', type=int, default=2)
    parser.add_argument('--feedback_eval_tile_size', type=int, default=384)
    parser.add_argument('--feedback_eval_tile_overlap', type=int, default=64)

    return parser.parse_args()


def main(args):
    pretrained_inputs = {
        '--base_ckpt': args.base_ckpt,
        '--spatial_ckpt': args.spatial_ckpt,
        '--st_ckpt': args.st_ckpt,
    }
    requested_pretrained = {
        name: value for name, value in pretrained_inputs.items() if value
    }
    if requested_pretrained:
        raise ValueError(
            'Scratch-only policy forbids pretrained initialization: %s'
            % requested_pretrained
        )
    if args.train_workers < 0 or args.val_workers < 0:
        raise ValueError('DataLoader worker counts must be non-negative.')
    if args.prefetch_factor <= 0:
        raise ValueError('prefetch_factor must be positive.')
    if (
        args.batch_size <= 0
        or args.gpu_num <= 0
        or args.gradient_accumulation_steps <= 0
    ):
        raise ValueError(
            'batch_size, gpu_num, and gradient_accumulation_steps '
            'must be positive.'
        )
    if args.eval_chunk_rows < 0:
        raise ValueError('eval_chunk_rows must be non-negative.')
    if args.eval_interval <= 0:
        raise ValueError('eval_interval must be positive.')
    if args.early_stopping_patience < 0:
        raise ValueError('early_stopping_patience must be non-negative.')
    if args.early_stopping_min_delta < 0:
        raise ValueError('early_stopping_min_delta must be non-negative.')
    if args.early_stopping_start_epoch < 1:
        raise ValueError('early_stopping_start_epoch must be positive.')
    if (
        args.early_stopping_patience > 0
        and args.early_stopping_start_epoch > args.epoch
    ):
        raise ValueError(
            'early_stopping_start_epoch cannot exceed --epoch when early '
            'stopping is enabled.'
        )
    if args.base_lr_mult <= 0:
        raise ValueError('base_lr_mult must be positive.')
    if args.mask_padded_frames and args.loss != 'f1_calibrated_ohem':
        raise ValueError(
            '--mask_padded_frames currently requires '
            '--loss f1_calibrated_ohem.'
        )
    if args.structure_bottleneck_channels <= 0:
        raise ValueError(
            'structure_bottleneck_channels must be positive.'
        )
    if args.structure_max_shift <= 0:
        raise ValueError('structure_max_shift must be positive.')
    if args.feedback_interval <= 0:
        raise ValueError('feedback_interval must be positive.')
    if args.feedback_alignment_levels < 2:
        raise ValueError('feedback_alignment_levels must be at least two.')
    if args.feedback_eval_tile_size <= 0:
        raise ValueError('feedback_eval_tile_size must be positive.')
    if not 0 <= args.feedback_eval_tile_overlap < args.feedback_eval_tile_size:
        raise ValueError('invalid feedback evaluation tile overlap.')

    runtime = initialize_distributed(args.gpu, args.gpu_num)
    if args.batch_size % runtime.world_size:
        finalize_distributed(runtime)
        raise ValueError(
            'Global --batch_size=%d must be divisible by world size %d.'
            % (args.batch_size, runtime.world_size)
        )
    seed_everything(
        args.seed + runtime.rank,
        deterministic=bool(args.deterministic),
    )

    if args.log_dir is None and runtime.is_main:
        timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        args.log_dir = (
            args.dataset + '__' + timestr + '__'
            + loss_experiment_name(args.loss) + '_'
            + args.model + '_DataL' + str(args.seqlen)
        )
    args.log_dir = broadcast_object(args.log_dir, runtime)
    args.savepath = str(Path(args.savepath).expanduser().resolve())
    experiment_root = Path(args.savepath) / 'sem_seg'
    experiment_dir = (experiment_root / args.log_dir).resolve()
    try:
        experiment_dir.relative_to(experiment_root)
    except ValueError as error:
        finalize_distributed(runtime)
        raise ValueError(
            '--log_dir must name an experiment under %s.' % experiment_root
        ) from error
    checkpoints_dir = experiment_dir / 'checkpoints'
    log_dir = experiment_dir / 'logs'

    experiment_preexisting = False
    resume_path = None
    setup_error = None
    if runtime.is_main:
        try:
            experiment_preexisting = (
                experiment_dir.exists() and any(experiment_dir.iterdir())
            )
            if args.resume_checkpoint:
                candidate = Path(args.resume_checkpoint).expanduser().resolve()
                if not candidate.is_file():
                    raise FileNotFoundError(
                        'Resume checkpoint does not exist: %s' % candidate
                    )
                resume_path = str(candidate)
            elif args.resume == 'auto':
                for filename in ('latest_model.pth', 'best_model.pth'):
                    candidate = checkpoints_dir / filename
                    if candidate.is_file():
                        resume_path = str(candidate)
                        break
                if resume_path is None and experiment_preexisting:
                    raise RuntimeError(
                        'Experiment directory is non-empty but has no resumable '
                        'checkpoint: %s. Refusing to overwrite it.'
                        % experiment_dir
                    )
            elif experiment_preexisting:
                raise RuntimeError(
                    'Experiment directory already contains files: %s. '
                    '--resume never refuses to overwrite them.' % experiment_dir
                )

            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            setup_error = '%s: %s' % (type(error).__name__, error)
    setup_error = broadcast_object(setup_error, runtime)
    resume_path = broadcast_object(resume_path, runtime)
    if setup_error is not None:
        finalize_distributed(runtime)
        raise RuntimeError(setup_error)
    distributed_barrier(runtime)

    logger = logging.getLogger('Model-rank%d' % runtime.rank)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    if runtime.is_main:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler = logging.FileHandler(log_dir / ('%s.txt' % args.model))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    def log_string(message):
        if runtime.is_main:
            logger.info(message)
            print(message, flush=True)

    def release_cuda_memory(stage):
        gc.collect()
        torch.cuda.synchronize(runtime.device)
        torch.cuda.empty_cache()
        if runtime.is_main:
            log_string(
                'CUDA memory after %s: allocated=%.3f GiB, reserved=%.3f GiB'
                % (
                    stage,
                    torch.cuda.memory_allocated(runtime.device) / (1024 ** 3),
                    torch.cuda.memory_reserved(runtime.device) / (1024 ** 3),
                )
            )

    log_string('PARAMETER ...')
    log_string(args)
    swanlab_run = None
    swanlab_module = None
    if args.use_swanlab and runtime.is_main:
        try:
            import swanlab as swanlab_module
        except ImportError as error:
            finalize_distributed(runtime)
            raise RuntimeError(
                'SwanLab is required when --use_swanlab=1.'
            ) from error
        try:
            swanlab_run = swanlab_module.init(
                project=args.swanlab_project,
                workspace=args.swanlab_workspace,
                experiment_name=args.log_dir,
                group=args.swanlab_group,
                logdir=str(experiment_dir / 'swanlog'),
                mode=args.swanlab_mode,
                id=args.swanlab_id,
                resume=args.swanlab_resume,
                config=vars(args),
            )
        except Exception as error:
            finalize_distributed(runtime)
            raise RuntimeError(
                'SwanLab initialization failed; refusing to train without logging: %s'
                % error
            ) from error

    args.datapath = str(Path(args.datapath).expanduser().resolve())
    root = args.datapath
    NUM_CLASSES = 1
    SEQ_LEN = args.seqlen
    local_batch_size = args.batch_size // runtime.world_size
    train_workers = (
        0 if args.train_workers == 0 else
        max(1, (args.train_workers + runtime.world_size - 1) // runtime.world_size)
    )

    log_string("start loading training data ...")
    train_transform = (
        SequenceGeometryAugmentation()
        if args.sequence_augmentation else None
    )
    TRAIN_DATASET = TrainIRSeqDataLoader(args.dataset, data_root=root, seq_len=SEQ_LEN, sample_rate=args.sample_rate,
                                         patch_size=args.patch_size, transform=train_transform)  # sample_rate=0.1, 0.03, 0.05
    train_sampler = None
    if runtime.distributed:
        train_sampler = DistributedSampler(
            TRAIN_DATASET,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed + runtime.rank)

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=local_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        **multiprocessing_loader_options(
            train_workers,
            args.prefetch_factor,
        )
    )
    if len(trainDataLoader) == 0:
        finalize_distributed(runtime)
        raise RuntimeError(
            'Training DataLoader has no full global batch; increase data or '
            'reduce --batch_size.'
        )

    TEST_DATASET = None
    sequence_datasets = None
    validationDataLoader = None
    if runtime.is_main:
        log_string("start loading validation data ...")
    TEST_DATASET = TestIRSeqDataLoader(
        args.dataset,
        data_root=root,
        seq_len=SEQ_LEN,
        cat_len=int(SEQ_LEN * 0.1),
        transform=None,
    )
    sequence_datasets = [
        TEST_DATASET[index]
        for index in range(runtime.rank, len(TEST_DATASET), runtime.world_size)
    ]
    if not sequence_datasets:
        finalize_distributed(runtime)
        raise RuntimeError('Validation shard contains no sequences.')
    validationDataLoader = torch.utils.data.DataLoader(
        ConcatDataset(sequence_datasets),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        **multiprocessing_loader_options(
            args.val_workers,
            args.prefetch_factor,
        )
    )

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    if runtime.is_main:
        log_string("The number of test data is: %d sequences" % len(TEST_DATASET))
        log_string(
            "DDP world_size=%d, global_batch=%d, per_rank_batch=%d; "
            "gradient_accumulation=%d, effective_batch=%d; "
            "DataLoader workers per rank=%d, validation=%d"
            % (
                runtime.world_size,
                args.batch_size,
                local_batch_size,
                args.gradient_accumulation_steps,
                args.batch_size * args.gradient_accumulation_steps,
                train_workers,
                args.val_workers,
            )
        )
        log_string(
            'Validation interval=%d; validation is sharded across %d rank(s).'
            % (args.eval_interval, runtime.world_size)
        )

    '''MODEL LOADING'''
    models_dir = (Path(ROOT_DIR) / 'networks' / 'models').resolve()
    model_source = (models_dir / ('%s.py' % args.model)).resolve()
    if model_source.parent != models_dir or not model_source.is_file():
        finalize_distributed(runtime)
        raise ValueError('Unknown or unsafe model name: %s' % args.model)
    MODEL = importlib.import_module(args.model)
    if runtime.is_main:
        model_snapshot = experiment_dir / model_source.name
        loss_source = (
            Path(ROOT_DIR) / 'networks' / 'losses' / 'segmentation_losses.py'
        )
        loss_snapshot = experiment_dir / loss_source.name
        if not model_snapshot.exists():
            shutil.copy2(model_source, model_snapshot)
        if not loss_snapshot.exists():
            shutil.copy2(loss_source, loss_snapshot)
        adapter_filenames = {
            'DeepPro-Plus_BRTD': ('brtd_adapter.py',),
            'DeepPro-Plus_BRTD2': ('brtd_v2_adapter.py',),
            'DeepPro-Plus_BRTD3': ('structure_adapters.py',),
            'DeepPro-FeedbackSTS': ('feedback_sts.py',),
        }.get(args.model, ())
        for adapter_filename in adapter_filenames:
            adapter_source = (
                Path(ROOT_DIR) / 'networks' / 'layers'
                / adapter_filename
            )
            adapter_snapshot = experiment_dir / adapter_source.name
            if not adapter_snapshot.exists():
                shutil.copy2(adapter_source, adapter_snapshot)
    distributed_barrier(runtime)

    config = model_configuration(MODEL, args)
    detector = MODEL.detector(NUM_CLASSES, SEQ_LEN, SEQ_LEN, **config)

    if args.base_ckpt and resume_path is None:
        base_checkpoint_path = Path(args.base_ckpt).expanduser().resolve()
        if not base_checkpoint_path.is_file():
            finalize_distributed(runtime)
            raise FileNotFoundError(
                'Base checkpoint does not exist: %s' % base_checkpoint_path
            )
        base_checkpoint = load_checkpoint(
            base_checkpoint_path,
            map_location='cpu',
        )
        base_state_dict = base_checkpoint.get(
            'model_state_dict',
            base_checkpoint,
        )
        incompatible = detector.load_state_dict(
            clean_model_state_dict(base_state_dict),
            strict=False,
        )
        allowed_missing_prefixes = ('brtd.',)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            finalize_distributed(runtime)
            raise RuntimeError(
                'Base checkpoint is incompatible with %s. Missing: %s; '
                'unexpected: %s'
                % (
                    args.model,
                    invalid_missing,
                    incompatible.unexpected_keys,
                )
            )
        log_string(
            'Initialized %s backbone from %s; new adapter keys: %d'
            % (
                args.model,
                base_checkpoint_path,
                len(incompatible.missing_keys),
            )
        )
        del base_checkpoint, base_state_dict, incompatible
    elif args.base_ckpt and resume_path is not None:
        log_string(
            'Resume checkpoint takes precedence over --base_ckpt; '
            'backbone initialization was skipped.'
        )
    elif resume_path is None:
        log_string(
            'Initialized %s from random weights; no base checkpoint loaded.'
            % args.model
        )

    detector = detector.to(runtime.device)
    if runtime.distributed:
        detector = DistributedDataParallel(
            detector,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
        )
    criterion = build_segmentation_loss(
        args.loss,
        eps=args.loss_eps,
        sls_eps=args.sls_eps,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tversky_fp_weight=args.tversky_fp_weight,
        tversky_fn_weight=args.tversky_fn_weight,
        tversky_gamma=args.tversky_gamma,
        bce_weight=args.bce_weight,
        hard_negative_topk=args.hard_negative_topk,
        hard_focal_weight=args.hard_focal_weight,
        sls_location_weight=args.sls_location_weight,
        sls_warmup_epochs=args.sls_warmup_epochs,
        tda_weight=args.tda_weight,
        tda_mean_size=args.tda_mean_size,
        tda_mean_contrast=args.tda_mean_contrast,
        tda_dilation=args.tda_dilation,
        stc_center_weight=args.stc_center_weight,
        stc_temporal_weight=args.stc_temporal_weight,
        stc_warmup_epochs=args.stc_warmup_epochs,
        f1_ohem_dice_weight=args.f1_ohem_dice_weight,
        f1_ohem_hard_weight=args.f1_ohem_hard_weight,
        f1_ohem_negative_ratio=args.f1_ohem_negative_ratio,
        f1_ohem_min_negatives=args.f1_ohem_min_negatives,
        f1_ohem_margin=args.f1_ohem_margin,
        f1_ohem_warmup_epochs=args.f1_ohem_warmup_epochs,
        f1_ohem_ramp_epochs=args.f1_ohem_ramp_epochs,
    ).to(runtime.device)
    log_string('Loss: %s - %s' % (args.loss, LOSS_DESCRIPTIONS[args.loss]))
    if getattr(criterion, 'requires_images', False):
        log_string(
            'WARNING: %s performs CPU connected-component extraction and '
            'will be slower than GPU-only losses.' % args.loss
        )

    if 'BRTD' in args.model:
        base_parameters = []
        adapter_parameters = []
        for name, parameter in detector.named_parameters():
            if not parameter.requires_grad:
                continue
            if 'brtd.' in name:
                adapter_parameters.append(parameter)
            else:
                base_parameters.append(parameter)
        if not adapter_parameters:
            finalize_distributed(runtime)
            raise RuntimeError(
                '%s was selected but no BRTD parameters were found.' % args.model
            )
        parameter_groups = [
            {
                'params': base_parameters,
                'lr': args.learning_rate * args.base_lr_mult,
                'lr_scale': args.base_lr_mult,
            },
            {
                'params': adapter_parameters,
                'lr': args.learning_rate,
                'lr_scale': 1.0,
            },
        ]
        log_string(
            'Optimizer groups: %d backbone tensors at %.3fx LR; '
            '%d BRTD tensors at 1.000x LR.'
            % (
                len(base_parameters),
                args.base_lr_mult,
                len(adapter_parameters),
            )
        )
    else:
        parameter_groups = filter(
            lambda parameter: parameter.requires_grad,
            detector.parameters(),
        )

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=args.learning_rate,
            momentum=0.9
        )
    grad_scaler = torch.cuda.amp.GradScaler(
        enabled=bool(args.train_amp)
    )

    best_iou = 0
    start_epoch = 0
    early_stopping_state = None
    if args.early_stopping_patience > 0:
        early_stopping_state = new_early_stopping_state(
            args.early_stopping_metric
        )
    if resume_path is not None:
        try:
            checkpoint = load_checkpoint(resume_path, map_location='cpu')
            checkpoint_model = checkpoint.get('model_name')
            if checkpoint_model is not None and checkpoint_model != args.model:
                raise ValueError(
                    'Checkpoint model %s does not match --model %s.'
                    % (checkpoint_model, args.model)
                )
            unwrap_model(detector).load_state_dict(
                checkpoint['model_state_dict'], strict=True
            )
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            move_optimizer_state(optimizer, runtime.device)
            saved_grad_scaler = checkpoint.get('grad_scaler_state_dict')
            if args.train_amp and saved_grad_scaler:
                grad_scaler.load_state_dict(saved_grad_scaler)
            start_epoch = int(checkpoint['epoch']) + 1
            best_iou = float(checkpoint.get('class_avg_iou', 0.0))
            saved_early_stopping_state = checkpoint.get(
                'early_stopping_state'
            )
            if (
                early_stopping_state is not None
                and saved_early_stopping_state is not None
            ):
                saved_metric = saved_early_stopping_state.get('metric')
                if saved_metric != args.early_stopping_metric:
                    raise ValueError(
                        'Checkpoint early-stopping metric %s does not match '
                        '--early_stopping_metric %s.'
                        % (saved_metric, args.early_stopping_metric)
                    )
                early_stopping_state.update(saved_early_stopping_state)
                early_stopping_state['stopped'] = False
            log_string(
                'Resumed checkpoint %s at epoch %d.'
                % (resume_path, start_epoch)
            )
            if early_stopping_state is not None:
                log_string(
                    'Restored early stopping: metric=%s best=%s at epoch %d; '
                    'bad_epochs=%d.'
                    % (
                        early_stopping_state['metric'],
                        early_stopping_state['best_value'],
                        early_stopping_state['best_epoch'],
                        early_stopping_state['bad_epochs'],
                    )
                )
            del checkpoint
        except Exception as error:
            finalize_distributed(runtime)
            raise RuntimeError(
                'Checkpoint recovery failed; refusing to overwrite experiment '
                '%s: %s' % (experiment_dir, error)
            ) from error
    else:
        log_string('Starting a new experiment from scratch.')
    log_string(
        'Training precision: %s.'
        % ('AMP FP16 model / FP32 loss' if args.train_amp else 'FP32')
    )

    if early_stopping_state is None:
        log_string('Early stopping disabled.')
    else:
        log_string(
            'Early stopping enabled: metric=%s mode=%s patience=%d '
            'min_delta=%.8f start_epoch=%d.'
            % (
                early_stopping_state['metric'],
                early_stopping_state['mode'],
                args.early_stopping_patience,
                args.early_stopping_min_delta,
                args.early_stopping_start_epoch,
            )
        )

    release_cuda_memory('checkpoint loading')


    LEARNING_RATE_CLIP = 1e-5
    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d/%s ****' % (epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        metric_counts = torch.zeros(3, device=runtime.device, dtype=torch.int64)
        loss_stats = torch.zeros(2, device=runtime.device, dtype=torch.float64)
        component_names = tuple(getattr(criterion, 'component_names', ()))
        component_stats = torch.zeros(
            len(component_names) + 1,
            device=runtime.device,
            dtype=torch.float64,
        )
        detector.train()
        optimizer.zero_grad(set_to_none=True)
        training_batch_count = len(trainDataLoader)

        for batch_index, (images, targets) in enumerate(tqdm(
            trainDataLoader,
            total=len(trainDataLoader),
            smoothing=0.9,
            disable=not runtime.is_main,
        )):
            valid_frames = None
            if args.mask_padded_frames:
                valid_frames = images.abs().sum(dim=(1, 3, 4)).ne(0)
                if not torch.all(valid_frames.any(dim=1)):
                    raise RuntimeError(
                        'A training clip contains no valid frames.'
                    )
                valid_frames = valid_frames.to(
                    runtime.device, non_blocking=True
                )
            images = images.float().to(runtime.device, non_blocking=True)
            targets = targets.float().to(runtime.device, non_blocking=True)

            accumulation_window_start = (
                batch_index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            accumulation_window_size = min(
                args.gradient_accumulation_steps,
                training_batch_count - accumulation_window_start,
            )
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == training_batch_count
            )
            synchronization_context = (
                nullcontext()
                if should_step
                or not isinstance(detector, DistributedDataParallel)
                else detector.no_sync()
            )
            with synchronization_context:
                with torch.autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                    enabled=bool(args.train_amp),
                ):
                    sequence_features, seq_midpred = detector(images)
                del sequence_features

                criterion_arguments = {
                    'images': images,
                    'epoch': epoch,
                }
                if valid_frames is not None:
                    criterion_arguments['valid_frames'] = valid_frames
                # Keep overlap reductions and hard-negative ranking in FP32;
                # only the compute-heavy network runs under autocast.
                loss = criterion(
                    seq_midpred.float(), targets, **criterion_arguments
                )
                grad_scaler.scale(
                    loss / accumulation_window_size
                ).backward()
            if should_step:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                midpred_choice = torch.sigmoid(seq_midpred.detach()).gt(
                    args.threshold_eval
                )
                batch_label = targets.gt(0)
                if valid_frames is not None:
                    metric_mask = valid_frames[:, :, None, None]
                    midpred_choice = torch.logical_and(
                        midpred_choice, metric_mask
                    )
                    batch_label = torch.logical_and(
                        batch_label, metric_mask
                    )
                metric_counts[0] += torch.logical_and(
                    midpred_choice,
                    batch_label,
                ).sum(dtype=torch.int64)
                metric_counts[1] += midpred_choice.sum(
                    dtype=torch.int64
                )
                metric_counts[2] += batch_label.sum(
                    dtype=torch.int64
                )
                loss_stats[0] += loss.detach().to(torch.float64)
                loss_stats[1] += 1
                for component_index, component_name in enumerate(component_names):
                    component_value = criterion.last_components[component_name]
                    component_stats[component_index] += component_value.to(
                        device=runtime.device, dtype=torch.float64
                    )
                component_stats[-1] += 1
            del images, targets, seq_midpred, loss, midpred_choice, batch_label
            del criterion_arguments, valid_frames

        optimizer.zero_grad(set_to_none=True)
        all_reduce_sum(metric_counts, runtime)
        all_reduce_sum(loss_stats, runtime)
        all_reduce_sum(component_stats, runtime)
        train_loss = (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
        train_iou, train_precision, train_recall, train_f1 = (
            binary_segmentation_metrics(*metric_counts)
        )
        component_count = component_stats[-1].clamp_min(1.0)
        component_means = {
            name: (component_stats[index] / component_count).item()
            for index, name in enumerate(component_names)
        }

        del metric_counts, loss_stats, component_stats
        release_cuda_memory('training cleanup')

        log_string('Training mean loss: %f' % train_loss)
        log_string('Training accuracy (IoU) of prediction: %f' % train_iou)
        log_string('Training pixel precision: %f' % train_precision)
        log_string('Training pixel recall: %f' % train_recall)
        log_string('Training pixel F1: %f' % train_f1)
        for component_name, component_value in component_means.items():
            log_string(
                'Training loss component %s: %f'
                % (component_name, component_value)
            )
        if swanlab_run is not None:
            swanlab_metrics = {
                'train/loss': train_loss,
                'train/iou': train_iou,
                'train/precision': train_precision,
                'train/recall': train_recall,
                'train/f1': train_f1,
                'train/lr': lr,
            }
            swanlab_metrics.update({
                'train/loss_component/' + name: value
                for name, value in component_means.items()
            })
            swanlab_module.log(swanlab_metrics, step=epoch + 1)

        should_evaluate = (
            (epoch + 1) % args.eval_interval == 0
            or epoch + 1 == args.epoch
        )
        if not should_evaluate:
            distributed_barrier(runtime)
            if runtime.is_main:
                state = make_checkpoint_state(
                    detector,
                    optimizer,
                    grad_scaler,
                    epoch,
                    best_iou,
                    args,
                    config,
                    early_stopping_state=early_stopping_state,
                )
                latest_path = checkpoints_dir / 'latest_model.pth'
                atomic_torch_save(state, latest_path)
                del state
                log_string(
                    'Skipped full validation at epoch %d '
                    '(eval_interval=%d); saved recoverable checkpoint.'
                    % (epoch + 1, args.eval_interval)
                )
            distributed_barrier(runtime)
            release_cuda_memory('validation skip cleanup')
            continue

        distributed_barrier(runtime)
        if runtime.is_main:
            log_string('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
        local_loss_sum, local_loss_count, local_metric_counts = (
            evaluate_sequences(
                unwrap_model(detector),
                criterion,
                sequence_datasets,
                validationDataLoader,
                runtime.device,
                args.threshold_eval,
                epoch,
                show_progress=runtime.is_main,
                use_amp=bool(args.eval_amp),
            )
        )
        eval_loss_stats = torch.tensor(
            [local_loss_sum, local_loss_count],
            device=runtime.device,
            dtype=torch.float64,
        )
        eval_metric_counts = local_metric_counts.to(
            device=runtime.device, non_blocking=True
        )
        all_reduce_sum(eval_loss_stats, runtime)
        all_reduce_sum(eval_metric_counts, runtime)
        eval_loss = (
            eval_loss_stats[0] / eval_loss_stats[1].clamp_min(1.0)
        ).item()
        eval_metrics = binary_segmentation_metrics(*eval_metric_counts)
        mIoU_mid, eval_precision, eval_recall, eval_f1 = eval_metrics
        del local_metric_counts, eval_loss_stats, eval_metric_counts

        stop_training = False
        if runtime.is_main:
            log_string('Eval mean loss: %f' % eval_loss)
            log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))
            log_string('Eval pixel precision: %f' % eval_precision)
            log_string('Eval pixel recall: %f' % eval_recall)
            log_string('Eval pixel F1: %f' % eval_f1)

            early_stopping_improved = False
            if early_stopping_state is not None:
                monitored_value = early_stopping_metric_value(
                    args.early_stopping_metric,
                    eval_loss,
                    mIoU_mid,
                    eval_f1,
                )
                early_stopping_improved, stop_training = (
                    update_early_stopping_state(
                        early_stopping_state,
                        monitored_value,
                        epoch + 1,
                        args.early_stopping_patience,
                        args.early_stopping_min_delta,
                        args.early_stopping_start_epoch,
                    )
                )
                log_string(
                    'Early stopping %s=%.6f; best=%.6f at epoch %d; '
                    'bad_epochs=%d/%d.'
                    % (
                        args.early_stopping_metric,
                        monitored_value,
                        early_stopping_state['best_value'],
                        early_stopping_state['best_epoch'],
                        early_stopping_state['bad_epochs'],
                        args.early_stopping_patience,
                    )
                )

            improved = mIoU_mid >= best_iou
            if mIoU_mid >= best_iou:
                best_iou = mIoU_mid
            state = make_checkpoint_state(
                detector,
                optimizer,
                grad_scaler,
                epoch,
                best_iou,
                args,
                config,
                early_stopping_state=early_stopping_state,
            )
            latest_path = checkpoints_dir / 'latest_model.pth'
            atomic_torch_save(state, latest_path)
            log_string('Saved recoverable checkpoint at %s' % latest_path)
            if (
                (epoch + 1) % 5 == 0
                or epoch + 1 == args.epoch
                or stop_training
                or early_stopping_improved
            ):
                epoch_path = checkpoints_dir / (
                    'epoch_%d_model.pth' % (epoch + 1)
                )
                atomic_torch_save(state, epoch_path)
                log_string('Saved epoch checkpoint at %s' % epoch_path)
            if improved:
                best_path = checkpoints_dir / 'best_model.pth'
                atomic_torch_save(state, best_path)
                log_string('Saved best checkpoint at %s' % best_path)
            if early_stopping_improved:
                early_best_path = (
                    checkpoints_dir / 'early_stopping_best_model.pth'
                )
                atomic_torch_save(state, early_best_path)
                log_string(
                    'Saved early-stopping best checkpoint at %s'
                    % early_best_path
                )
            del state
            log_string('Best mIoU_mid: %f' % best_iou)
            if swanlab_run is not None:
                eval_swanlab_metrics = {
                    'eval/loss': eval_loss,
                    'eval/iou': mIoU_mid,
                    'eval/precision': eval_precision,
                    'eval/recall': eval_recall,
                    'eval/f1': eval_f1,
                    'eval/best_iou': best_iou,
                }
                if early_stopping_state is not None:
                    eval_swanlab_metrics.update({
                        'eval/early_stopping_best': (
                            early_stopping_state['best_value']
                        ),
                        'eval/early_stopping_bad_epochs': (
                            early_stopping_state['bad_epochs']
                        ),
                    })
                swanlab_module.log(
                    eval_swanlab_metrics, step=epoch + 1
                )
            if stop_training:
                log_string(
                    'EARLY STOP triggered at epoch %d: %s did not improve '
                    'by at least %.8f for %d counted evaluations. Best '
                    'value %.6f was observed at epoch %d.'
                    % (
                        epoch + 1,
                        args.early_stopping_metric,
                        args.early_stopping_min_delta,
                        args.early_stopping_patience,
                        early_stopping_state['best_value'],
                        early_stopping_state['best_epoch'],
                    )
                )

        best_iou = broadcast_object(
            best_iou if runtime.is_main else None,
            runtime,
        )
        stop_training = broadcast_object(
            stop_training if runtime.is_main else None,
            runtime,
        )
        distributed_barrier(runtime)
        release_cuda_memory('evaluation cleanup')
        if stop_training:
            break

    if swanlab_run is not None and hasattr(swanlab_module, 'finish'):
        try:
            swanlab_module.finish()
        except Exception as error:
            # Cloud finalization must not invalidate a fully completed local
            # training run or prevent deterministic submission postprocessing.
            log_string(
                'WARNING: SwanLab finalization failed after training; '
                'continuing with local artifacts: %s' % error
            )

    del trainDataLoader, validationDataLoader
    del detector, criterion, optimizer
    release_cuda_memory('training shutdown')
    distributed_barrier(runtime)
    is_main = runtime.is_main
    finalize_distributed(runtime)
    return is_main


if __name__ == '__main__':
    args = parse_args()
    if launch_with_torchrun_if_needed(
        __file__, args.gpu, args.gpu_num
    ):
        raise SystemExit(0)
    run_followup_test = main(args)
    if run_followup_test and args.run_test_after_train:
        test_command = [
            sys.executable,
            str(Path(BASE_DIR) / 'test.py'),
            '--gpu',
            parse_visible_devices(args.gpu)[0],
            '--seqlen',
            str(args.seqlen),
            '--datapath',
            args.datapath,
            '--dataset',
            args.dataset,
            '--logpath',
            args.savepath,
            '--log_dir',
            args.log_dir,
            '--eval_chunk_rows',
            str(args.eval_chunk_rows),
        ]
        if args.eval_amp:
            test_command.append('--amp')
        subprocess.run(test_command, check=True, cwd=BASE_DIR)
