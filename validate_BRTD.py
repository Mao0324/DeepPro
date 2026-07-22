import argparse
import inspect
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_utils.TestDataLoader import TestIRSeqDataLoader
from sequence_utils import SequenceAccumulator, frame_range_length
from runtime_utils import atomic_torch_save, load_checkpoint, parse_visible_devices


def parse_args():
    parser = argparse.ArgumentParser('Isolated BRTD validation')
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--metrics_path', required=True)
    parser.add_argument('--gpu', type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--seqlen', type=int, default=40)
    parser.add_argument('--datapath', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--experiment_dir', required=True)
    parser.add_argument('--threshold_eval', type=float, default=0.5)
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1])
    parser.add_argument('--update_checkpoint', type=int, default=0, choices=[0, 1])
    parser.add_argument('--best_model_path', default='')
    return parser.parse_args()


def main(args):
    visible_devices = parse_visible_devices(args.gpu)
    if len(visible_devices) != 1:
        raise ValueError('validate_BRTD.py requires exactly one GPU.')
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices[0]
    experiment_dir = Path(args.experiment_dir).resolve()
    checkpoint = load_checkpoint(args.checkpoint_path, map_location='cpu')

    model_name = checkpoint.get('model_name')
    if model_name is None:
        log_files = sorted((experiment_dir / 'logs').glob('*.txt'))
        if len(log_files) != 1:
            raise RuntimeError('Unable to determine model name for validation.')
        model_name = log_files[0].stem

    model_path = (experiment_dir / ('%s.py' % model_name)).resolve()
    if model_path.parent != experiment_dir or not model_path.is_file():
        raise ValueError('Unsafe or missing model snapshot: %s' % model_name)
    sys.path.insert(0, str(experiment_dir))
    model_module = importlib.import_module(model_name)
    constructor_parameters = inspect.signature(model_module.detector).parameters
    model_config = {
        key: value for key, value in checkpoint.get('model_config', {}).items()
        if key in constructor_parameters
        and key not in {'spatial_ckpt', 'st_ckpt'}
    }
    if 'freeze_pretrained' in model_config:
        model_config['freeze_pretrained'] = False
    detector = model_module.detector(
        1,
        args.seqlen,
        args.seqlen,
        **model_config,
    ).cuda()
    state_dict = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in checkpoint['model_state_dict'].items()
    }
    detector.load_state_dict(state_dict)
    detector.eval()

    dataset = TestIRSeqDataLoader(
        args.dataset,
        data_root=args.datapath,
        seq_len=args.seqlen,
        cat_len=int(args.seqlen * 0.1),
        transform=None,
    )

    metric_counts = torch.zeros(2, device='cuda', dtype=torch.int64)
    eval_bar = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc='Isolated eval',
        ascii=True,
        dynamic_ncols=False,
        ncols=100,
        mininterval=0.5,
        file=sys.stdout,
    )

    with torch.inference_mode():
        for _, sequence_dataset in eval_bar:
            sequence_loader = torch.utils.data.DataLoader(
                sequence_dataset,
                batch_size=1,
                shuffle=False,
            )
            accumulator = SequenceAccumulator()
            for images, targets, _, first_end in sequence_loader:
                images = images.float().cuda(non_blocking=True)
                targets = targets.float().cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                    sequence_features, prediction = detector(images)
                    del sequence_features
                    if prediction.shape[-2:] != targets.shape[-2:]:
                        prediction = F.interpolate(
                            prediction,
                            size=targets.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                valid_length = frame_range_length(first_end)
                accumulator.add(
                    torch.sigmoid(prediction[:, :valid_length]),
                    targets[:, :valid_length],
                    first_end,
                )
                del images, targets, prediction

            binary_prediction = accumulator.predictions.gt(args.threshold_eval)
            labels = accumulator.targets.gt(0)
            metric_counts[0] += torch.logical_and(
                binary_prediction, labels
            ).sum(dtype=torch.int64)
            metric_counts[1] += torch.logical_or(
                binary_prediction, labels
            ).sum(dtype=torch.int64)
            del accumulator, binary_prediction, labels

    total_intersection, total_union = metric_counts.tolist()
    mean_iou = total_intersection / max(total_union, 1)
    metrics = {
        'iou': float(mean_iou),
        'intersection': float(total_intersection),
        'union': float(total_union),
    }
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    if args.update_checkpoint:
        previous_best = float(checkpoint.get('class_avg_iou', 0.0))
        is_best = mean_iou >= previous_best
        checkpoint['class_avg_iou'] = max(previous_best, float(mean_iou))
        atomic_torch_save(checkpoint, args.checkpoint_path)
        if is_best and args.best_model_path:
            atomic_torch_save(checkpoint, args.best_model_path)
    print('Isolated validation IoU: %.6f' % mean_iou)


if __name__ == '__main__':
    main(parse_args())
