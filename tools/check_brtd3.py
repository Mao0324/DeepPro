#!/usr/bin/env python3
"""Check every BRTD3 variant for pretrain identity and gradient flow."""

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from networks.layers.structure_adapters import STRUCTURE_VARIANTS
from runtime_utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint', type=Path,
        default=PROJECT_ROOT / 'pretrained'
        / 'SatVideoIRSDT_DeepPro-Plus_pretrained_init.pth',
    )
    parser.add_argument('--variant', choices=('all',) + STRUCTURE_VARIANTS,
                        default='all')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seqlen', type=int, default=40)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(
        args.batch_size, args.seqlen, args.height, args.width
    ) <= 0:
        raise ValueError('batch size and tensor dimensions must be positive')
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    base_module = importlib.import_module('networks.models.DeepPro-Plus')
    brtd3_module = importlib.import_module(
        'networks.models.DeepPro-Plus_BRTD3'
    )
    base = base_module.detector(
        1, args.seqlen, args.seqlen
    ).to(args.device).eval()
    base.load_state_dict(state_dict, strict=True)
    inputs = torch.randn(
        args.batch_size, 1, args.seqlen, args.height, args.width,
        device=args.device,
    )
    with torch.no_grad():
        base_logits = base(inputs)[1]
    del base

    variants = STRUCTURE_VARIANTS if args.variant == 'all' else (
        args.variant,
    )
    for variant in variants:
        model = brtd3_module.detector(
            1,
            args.seqlen,
            args.seqlen,
            structure_variant=variant,
            eval_chunk_rows=max(1, args.height // 2),
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith('brtd.')
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                '%s pretrain mismatch missing=%s unexpected=%s'
                % (variant, invalid_missing, incompatible.unexpected_keys)
            )
        model = model.to(args.device).eval()
        with torch.no_grad():
            _, logits, auxiliary = model(inputs, return_aux=True)
        maximum_error = (logits - base_logits).abs().max().item()
        if maximum_error != 0.0:
            raise RuntimeError(
                '%s zero-init prediction error is %.9g'
                % (variant, maximum_error)
            )
        if not auxiliary:
            raise RuntimeError('%s returned no diagnostics' % variant)

        model.train()
        model.zero_grad(set_to_none=True)
        prediction = model(inputs)[1]
        target = torch.zeros_like(prediction)
        target[:, :, args.height // 2, args.width // 2] = 1.0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, target
        )
        loss.backward()
        gradient_sum = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.brtd.named_parameters()
            if 'projection.weight' in name and parameter.grad is not None
        )
        if not gradient_sum > 0.0:
            raise RuntimeError(
                '%s has no residual-projection gradient' % variant
            )
        adapter_parameters = sum(
            parameter.numel() for parameter in model.brtd.parameters()
        )
        print(
            '%-18s PASS new_keys=%d adapter_params=%d '
            'loss=%.6f projection_grad=%.6f'
            % (
                variant,
                len(incompatible.missing_keys),
                adapter_parameters,
                float(loss.detach()),
                gradient_sum,
            )
        )
        del model, prediction, target, loss
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    print('BRTD3 sanity check passed for %d variant(s).' % len(variants))


if __name__ == '__main__':
    main()
