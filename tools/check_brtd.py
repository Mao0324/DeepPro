import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Sanity-check DeepPro-Plus_BRTD.')
    parser.add_argument('--seqlen', type=int, default=8)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    base_module = importlib.import_module('networks.models.DeepPro-Plus')
    brtd_module = importlib.import_module('networks.models.DeepPro-Plus_BRTD')

    base = base_module.detector(1, args.seqlen, args.seqlen)
    brtd = brtd_module.detector(1, args.seqlen, args.seqlen, zero_init=True)
    incompatible = brtd.load_state_dict(base.state_dict(), strict=False)

    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith('brtd.')
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            'Backbone state mismatch. Missing: %s; unexpected: %s'
            % (invalid_missing, incompatible.unexpected_keys)
        )

    base = base.to(args.device).eval()
    brtd = brtd.to(args.device).eval()
    inputs = torch.randn(
        1,
        1,
        args.seqlen,
        args.height,
        args.width,
        device=args.device,
    )

    with torch.no_grad():
        base_prediction = base(inputs)[1]
        _, brtd_prediction, auxiliary = brtd(inputs, return_aux=True)

    maximum_error = (base_prediction - brtd_prediction).abs().max().item()
    print('Missing BRTD keys:', len(incompatible.missing_keys))
    print('Initial maximum prediction error:', maximum_error)
    print('Temporal weights:', tuple(auxiliary['temporal_weights'].shape))
    print('Reference weights:', tuple(auxiliary['reference_weights'].shape))
    print('Reliability gate:', tuple(auxiliary['reliability_gate'].shape))

    if maximum_error > 1e-6:
        raise RuntimeError('Zero-initialized BRTD does not reproduce DeepPro-Plus.')

    brtd.train()
    target = torch.zeros_like(brtd_prediction)
    target[:, :, args.height // 2, args.width // 2] = 1
    prediction = brtd(inputs)[1]
    loss = brtd_module.SoftLoUloss()(prediction, target)
    loss.backward()

    projection_gradient = brtd.brtd.delta_projection.weight.grad
    if projection_gradient is None or projection_gradient.abs().sum().item() == 0:
        raise RuntimeError('No gradient reaches the BRTD residual projection.')

    print('Backward loss:', loss.item())
    print('Projection gradient sum:', projection_gradient.abs().sum().item())
    print('BRTD sanity check passed.')


if __name__ == '__main__':
    main()
