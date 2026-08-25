#!/usr/bin/env python3
"""Check every BRTD3 variant for scratch identity and gradient flow."""

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from networks.layers.structure_adapters import STRUCTURE_VARIANTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('all',) + STRUCTURE_VARIANTS,
                        default='all')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seqlen', type=int, default=40)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    return parser.parse_args()


def check_raw_apmd_padding(adapter, device, height, width, variant):
    """Check that loader padding cannot create appearance/motion residuals."""
    clip_length = 5
    total_length = clip_length * 2
    clip = torch.randn(
        1, 1, clip_length, height, width, device=device
    ) + 0.25
    raw_at_start = torch.zeros(
        1, 1, total_length, height, width, device=device
    )
    raw_at_end = torch.zeros_like(raw_at_start)
    raw_at_start[:, :, :clip_length] = clip
    raw_at_end[:, :, clip_length:] = clip
    features = torch.zeros(
        1, adapter.channels, total_length, height, width,
        device=device,
    )

    adapter.eval()
    with torch.no_grad():
        adapter.projection.weight.fill_(0.01)
        start_output, start_aux = adapter(
            features, raw_at_start, return_aux=True
        )
        end_output, end_aux = adapter(
            features, raw_at_end, return_aux=True
        )
    valid_error = (
        start_output[:, :, :clip_length]
        - end_output[:, :, clip_length:]
    ).abs().max().item()
    padding_response = max(
        start_output[:, :, clip_length:].abs().max().item(),
        end_output[:, :, :clip_length].abs().max().item(),
    )
    first_weight_error = (
        start_aux['first_order_scale_weights'].sum(dim=0) - 1.0
    ).abs().max().item()
    second_weight_error = (
        end_aux['second_order_scale_weights'].sum(dim=0) - 1.0
    ).abs().max().item()
    if valid_error > 1e-6:
        raise RuntimeError(
            'raw_apmd depends on padding side; max error %.9g'
            % valid_error
        )
    if padding_response != 0.0:
        raise RuntimeError(
            'raw_apmd generated residuals on padding: %.9g'
            % padding_response
        )
    if max(first_weight_error, second_weight_error) > 1e-6:
        raise RuntimeError('raw_apmd scale weights are not normalized')
    print(
        '%s padding PASS valid_shift_error=%.3g '
        'padding_response=%.3g'
        % (variant, valid_error, padding_response)
    )


def main():
    args = parse_args()
    if min(
        args.batch_size, args.seqlen, args.height, args.width
    ) <= 0:
        raise ValueError('batch size and tensor dimensions must be positive')
    base_module = importlib.import_module('networks.models.DeepPro-Plus')
    brtd3_module = importlib.import_module(
        'networks.models.DeepPro-Plus_BRTD3'
    )
    base = base_module.detector(
        1, args.seqlen, args.seqlen
    ).to(args.device).eval()
    # Compare with the same freshly randomized backbone. The check therefore
    # proves the adapter's zero-residual identity without loading any external
    # initialization weights.
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in base.state_dict().items()
    }
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
            eval_chunk_rows=0,
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith('brtd.')
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                '%s scratch-backbone mismatch missing=%s unexpected=%s'
                % (variant, invalid_missing, incompatible.unexpected_keys)
            )
        model = model.to(args.device).eval()
        with torch.no_grad():
            _, logits, auxiliary = model(inputs, return_aux=True)
        maximum_error = (logits - base_logits).abs().max().item()
        projection_init_scale = float(getattr(
            model.brtd, 'projection_init_scale', 0.0
        ))
        if projection_init_scale == 0.0 and maximum_error != 0.0:
            raise RuntimeError(
                '%s zero-init prediction error is %.9g'
                % (variant, maximum_error)
            )
        if projection_init_scale > 0.0 and maximum_error == 0.0:
            raise RuntimeError(
                '%s scaled projection produced an identity prediction'
                % variant
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
        upstream_gradient_sum = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in model.brtd.named_parameters()
            if 'projection.weight' not in name
            and parameter.grad is not None
        )
        if projection_init_scale > 0.0 and not upstream_gradient_sum > 0.0:
            raise RuntimeError(
                '%s has no first-step upstream adapter gradient' % variant
            )
        adapter_parameters = sum(
            parameter.numel() for parameter in model.brtd.parameters()
        )
        if variant.startswith('raw_apmd'):
            check_raw_apmd_padding(
                model.brtd, args.device, args.height, args.width,
                variant,
            )
        if 'hybrid_rms' in variant:
            hybrid_mixes = [
                module.channel_mix().detach().flatten()
                for module in model.brtd.modules()
                if isinstance(module, torch.nn.Module)
                and hasattr(module, 'channel_mix')
            ]
            if not hybrid_mixes:
                raise RuntimeError('hybrid_rms normalization was not instantiated')
            hybrid_mix_error = (
                torch.cat(hybrid_mixes) - 0.5
            ).abs().max().item()
            if hybrid_mix_error > 1e-7:
                raise RuntimeError(
                    'hybrid_rms initial channel mix error is %.9g'
                    % hybrid_mix_error
                )
        if 'multiscale_contrast' in variant:
            contrast_weight_error = (
                auxiliary['contrast_scale_weights'].sum(dim=0) - 1.0
            ).abs().max().item()
            if contrast_weight_error > 1e-6:
                raise RuntimeError(
                    'raw_apmd contrast weights are not normalized'
                )
        if 'scratch_bandpass' in variant:
            if 'temporal_bandpass' not in auxiliary:
                raise RuntimeError('%s returned no temporal bandpass' % variant)
        if 'scratch_detail' in variant:
            if 'backbone_detail' not in auxiliary:
                raise RuntimeError('%s returned no backbone detail' % variant)
        print(
            '%-18s PASS new_keys=%d adapter_params=%d init_scale=%.3f '
            'identity_error=%.3g loss=%.6f projection_grad=%.6f '
            'upstream_grad=%.6f'
            % (
                variant,
                len(incompatible.missing_keys),
                adapter_parameters,
                projection_init_scale,
                maximum_error,
                float(loss.detach()),
                gradient_sum,
                upstream_gradient_sum,
            )
        )
        del model, prediction, target, loss
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    print('BRTD3 sanity check passed for %d variant(s).' % len(variants))


if __name__ == '__main__':
    main()
