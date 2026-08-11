#!/usr/bin/env python3
"""Select saved checkpoints with the strongest validation pixel F1."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch


EPOCH_PATTERN = re.compile(r'\*{4} Epoch (\d+)/(\d+) \*{4}')
F1_PATTERN = re.compile(r'Eval pixel F1:\s*([0-9.eE+-]+)')
CHECKPOINT_PATTERN = re.compile(r'epoch_(\d+)_model\.pth$')


def load_checkpoint_epoch(path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    return int(checkpoint['epoch']) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-file', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()

    log_file = args.log_file.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if not log_file.is_file():
        parser.error('log file does not exist: %s' % log_file)
    if not checkpoint_dir.is_dir():
        parser.error('checkpoint directory does not exist: %s' % checkpoint_dir)
    if args.top_k < 1:
        parser.error('--top-k must be positive')

    epoch_f1 = {}
    current_epoch = None
    for line in log_file.read_text(encoding='utf-8').splitlines():
        epoch_match = EPOCH_PATTERN.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        f1_match = F1_PATTERN.search(line)
        if f1_match and current_epoch is not None:
            epoch_f1[current_epoch] = float(f1_match.group(1))

    saved_epochs = {}
    for path in checkpoint_dir.glob('epoch_*_model.pth'):
        match = CHECKPOINT_PATTERN.search(path.name)
        if match:
            saved_epochs[int(match.group(1))] = path
    ranked = sorted(
        (
            (score, epoch, saved_epochs[epoch])
            for epoch, score in epoch_f1.items()
            if epoch in saved_epochs
        ),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    if not ranked:
        raise RuntimeError('No saved epoch checkpoint has a parsed Eval pixel F1.')

    candidates = [
        {
            'selector': 'epoch:%d' % epoch,
            'epoch': epoch,
            'eval_pixel_f1': score,
            'checkpoint': str(path.resolve()),
        }
        for score, epoch, path in ranked[:args.top_k]
    ]

    best_path = checkpoint_dir / 'best_model.pth'
    if not best_path.is_file():
        raise FileNotFoundError(best_path)
    best_epoch = load_checkpoint_epoch(best_path)
    selected_epochs = {item['epoch'] for item in candidates}
    if best_epoch not in selected_epochs:
        candidates.append({
            'selector': 'best',
            'epoch': best_epoch,
            'eval_pixel_f1': epoch_f1.get(best_epoch),
            'checkpoint': str(best_path.resolve()),
        })

    payload = {
        'log_file': str(log_file),
        'checkpoint_dir': str(checkpoint_dir),
        'top_k_saved_epochs': args.top_k,
        'candidates': candidates,
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    temporary_path.replace(output_path)

    print(
        'Selected checkpoint candidates: %s'
        % ', '.join(
            '%s(F1=%s)' % (item['selector'], item['eval_pixel_f1'])
            for item in candidates
        ),
        file=sys.stderr,
    )
    for item in candidates:
        print(item['selector'])


if __name__ == '__main__':
    main()
