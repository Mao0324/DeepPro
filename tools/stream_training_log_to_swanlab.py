#!/usr/bin/env python3
"""Stream completed epochs from an existing DeepPro log to SwanLab."""

import argparse
import json
import re
import time
from pathlib import Path

import swanlab


FIELD_PATTERNS = {
    'train/lr': r'^Learning rate:([0-9.eE+-]+)$',
    'train/loss': r'^Training mean loss: ([0-9.eE+-]+)$',
    'train/iou': r'^Training accuracy \(IoU\) of prediction: ([0-9.eE+-]+)$',
    'train/precision': r'^Training pixel precision: ([0-9.eE+-]+)$',
    'train/recall': r'^Training pixel recall: ([0-9.eE+-]+)$',
    'train/f1': r'^Training pixel F1: ([0-9.eE+-]+)$',
    'eval/loss': r'^Eval mean loss: ([0-9.eE+-]+)$',
    'eval/iou': r'^Eval avg class IoU of prediction: ([0-9.eE+-]+)$',
    'eval/precision': r'^Eval pixel precision: ([0-9.eE+-]+)$',
    'eval/recall': r'^Eval pixel recall: ([0-9.eE+-]+)$',
    'eval/f1': r'^Eval pixel F1: ([0-9.eE+-]+)$',
    'eval/best_iou': r'^Best mIoU_mid: ([0-9.eE+-]+)$',
}
REQUIRED_FIELDS = frozenset(FIELD_PATTERNS)
EPOCH_PATTERN = re.compile(r'^\*\*\*\* Epoch (\d+)/(\d+) \*\*\*\*$', re.M)
COMPONENT_PATTERN = re.compile(
    r'^Training loss component ([^:]+): ([0-9.eE+-]+)$', re.M
)
EARLY_STOP_PATTERN = re.compile(
    r'^Early stopping [^=]+=([0-9.eE+-]+); best=([0-9.eE+-]+) '
    r'at epoch \d+; bad_epochs=(\d+)/\d+\.$', re.M
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-file', type=Path, required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--state-file', type=Path, required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--init-mode', choices=('pretrained', 'scratch'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--min-epoch', type=int, default=9)
    parser.add_argument('--poll-seconds', type=float, default=20.0)
    return parser.parse_args()


def load_state(path):
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {int(epoch) for epoch in payload.get('uploaded_epochs', [])}


def save_state(path, uploaded_epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps({'uploaded_epochs': sorted(uploaded_epochs)}, indent=2),
        encoding='utf-8',
    )
    temporary.replace(path)


def completed_epochs(text, min_epoch):
    text = re.sub(r'^.* - INFO - ', '', text, flags=re.M)
    matches = list(EPOCH_PATTERN.finditer(text))
    completed = []
    for index, match in enumerate(matches):
        epoch = int(match.group(1))
        if epoch < min_epoch:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        metrics = {}
        for name, pattern in FIELD_PATTERNS.items():
            value_match = re.search(pattern, block, re.M)
            if value_match:
                metrics[name] = float(value_match.group(1))
        if not REQUIRED_FIELDS.issubset(metrics):
            continue
        for component_name, value in COMPONENT_PATTERN.findall(block):
            metrics['train/loss_component/' + component_name] = float(value)
        early_stop = EARLY_STOP_PATTERN.search(block)
        if early_stop:
            metrics['eval/early_stopping_best'] = float(early_stop.group(2))
            metrics['eval/early_stopping_bad_epochs'] = int(early_stop.group(3))
        completed.append((epoch, metrics))
    return completed


def status_is_terminal(path):
    if not path.is_file():
        return False
    first_line = path.read_text(encoding='utf-8', errors='replace').splitlines()
    return bool(first_line) and first_line[0].startswith(('COMPLETE', 'FAILED'))


def main():
    args = parse_args()
    if args.min_epoch < 1 or args.poll_seconds <= 0:
        raise ValueError('min epoch and poll interval must be positive')
    uploaded = load_state(args.state_file)
    run = swanlab.init(
        project=args.project,
        experiment_name=args.run_name,
        group=args.group,
        logdir=str(args.state_file.parent / 'swanlog'),
        mode='cloud',
        id=args.run_id,
        resume='allow',
        config={
            'structure_variant': args.variant,
            'initialization': args.init_mode,
            'seed': args.seed,
            'late_logging_min_epoch': args.min_epoch,
            'source_log': str(args.log_file),
        },
    )
    print('SwanLab sidecar started; first eligible epoch=%d' % args.min_epoch, flush=True)
    try:
        while True:
            if args.log_file.is_file():
                text = args.log_file.read_text(encoding='utf-8', errors='replace')
                for epoch, metrics in completed_epochs(text, args.min_epoch):
                    if epoch in uploaded:
                        continue
                    swanlab.log(metrics, step=epoch)
                    uploaded.add(epoch)
                    save_state(args.state_file, uploaded)
                    print('Uploaded completed epoch %d' % epoch, flush=True)
            if status_is_terminal(args.status_file):
                print('Training pipeline reached terminal status.', flush=True)
                break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == '__main__':
    main()
