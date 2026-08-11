"""Zero-initialized structural adapters for DeepPro-Plus ablations.

All modules preserve ``[B,C,T,H,W]`` shape and keep an explicit appearance
path.  Their final projection is zero initialized so a pretrained
DeepPro-Plus model produces identical logits immediately after insertion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


STRUCTURE_VARIANTS = (
    'second_order',
    'lfp_shallow',
    'lfp_deep',
    'global_align',
    'local_align',
    'multiscale_head',
    'bidirectional',
    'tdc_dual_stream',
)


def _group_count(channels, maximum=4):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _zero_projection(in_channels, out_channels):
    projection = nn.Conv3d(
        in_channels, out_channels, kernel_size=1, bias=False
    )
    nn.init.zeros_(projection.weight)
    return projection


def _local_contrast(x):
    near = F.avg_pool3d(
        x, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
    )
    wide = F.avg_pool3d(
        x, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3)
    )
    return x - 0.5 * (near + wide)


def _temporal_neighbours(x, step):
    if step <= 0 or step >= x.shape[2]:
        raise ValueError(
            'temporal step must be in [1,T), got %d for T=%d'
            % (step, x.shape[2])
        )
    first = x[:, :, :1].expand(-1, -1, step, -1, -1)
    last = x[:, :, -1:].expand(-1, -1, step, -1, -1)
    previous = torch.cat([first, x[:, :, :-step]], dim=2)
    following = torch.cat([x[:, :, step:], last], dim=2)
    return previous, following


def _reduction(channels, bottleneck_channels):
    return nn.Sequential(
        nn.Conv3d(channels, bottleneck_channels, kernel_size=1, bias=False),
        nn.GroupNorm(
            _group_count(bottleneck_channels), bottleneck_channels
        ),
        nn.SiLU(inplace=True),
    )


def _fusion(in_channels, bottleneck_channels):
    return nn.Sequential(
        nn.Conv3d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        ),
        nn.GroupNorm(
            _group_count(bottleneck_channels), bottleneck_channels
        ),
        nn.SiLU(inplace=True),
    )


def _reliability_gate(bottleneck_channels, input_multiplier=2):
    gate = nn.Sequential(
        nn.Conv3d(
            bottleneck_channels * input_multiplier,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        ),
        nn.GroupNorm(
            _group_count(bottleneck_channels), bottleneck_channels
        ),
        nn.SiLU(inplace=True),
        nn.Conv3d(bottleneck_channels, 1, kernel_size=1),
        nn.Sigmoid(),
    )
    nn.init.zeros_(gate[-2].weight)
    nn.init.constant_(gate[-2].bias, -2.0)
    return gate


class SecondOrderMotionAdapter(nn.Module):
    """Fuse appearance with first- and second-order temporal anomalies."""

    def __init__(self, channels=32, bottleneck_channels=8):
        super().__init__()
        self.reduce = _reduction(channels, bottleneck_channels)
        steps = (1, 2, 4)
        self.steps = steps
        self.motion_fusion = _fusion(
            bottleneck_channels * len(steps) * 2,
            bottleneck_channels,
        )
        self.fusion = _fusion(bottleneck_channels * 3, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        motion_parts = []
        for step in self.steps:
            previous, following = _temporal_neighbours(appearance, step)
            first_order = 0.5 * (following - previous)
            second_order = following - 2.0 * appearance + previous
            motion_parts.extend([first_order, second_order])
        motion = self.motion_fusion(torch.cat(motion_parts, dim=1))
        contrast = _local_contrast(appearance)
        fused = self.fusion(torch.cat([appearance, motion, contrast], dim=1))
        reliability = self.gate(
            torch.cat([motion.abs(), contrast.abs()], dim=1)
        )
        delta = self.projection(fused)
        output = x + reliability * delta
        if not return_aux:
            return output
        return output, {
            'motion': motion,
            'local_contrast': contrast,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


class LowFrequencyPurificationAdapter(nn.Module):
    """Low-frequency-guided high-frequency purification inspired by LFP."""

    def __init__(self, channels=32, bottleneck_channels=8):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.reduce = _reduction(channels, bottleneck_channels)
        coefficients = torch.tensor(
            [1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32
        )
        kernel = torch.outer(coefficients, coefficients)
        kernel = kernel / kernel.sum()
        self.register_buffer(
            'gaussian_kernel', kernel.view(1, 1, 1, 5, 5)
        )
        self.attention = nn.Sequential(
            nn.Conv3d(
                2, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = _fusion(bottleneck_channels * 3, bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def _gaussian_blur(self, x):
        kernel = self.gaussian_kernel.to(dtype=x.dtype).expand(
            self.bottleneck_channels, 1, 1, 5, 5
        )
        return F.conv3d(
            x, kernel, padding=(0, 2, 2), groups=self.bottleneck_channels
        )

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        low_frequency = self._gaussian_blur(appearance)
        high_frequency = appearance - low_frequency
        low_statistics = torch.cat([
            low_frequency.mean(dim=1, keepdim=True),
            low_frequency.amax(dim=1, keepdim=True),
        ], dim=1)
        attention = self.attention(low_statistics)
        purified_high = self._gaussian_blur(attention * high_frequency)
        fused = self.fusion(torch.cat([
            appearance, low_frequency, purified_high
        ], dim=1))
        delta = self.projection(fused)
        output = x + attention * delta
        if not return_aux:
            return output
        return output, {
            'low_frequency': low_frequency,
            'purified_high_frequency': purified_high,
            'frequency_attention': attention,
            'residual_delta': delta,
        }


class _AlignmentAdapter(nn.Module):
    def __init__(
        self,
        channels=32,
        bottleneck_channels=8,
        max_shift=4.0,
        dense_flow=False,
    ):
        super().__init__()
        if max_shift <= 0:
            raise ValueError('max_shift must be positive')
        self.max_shift = float(max_shift)
        self.dense_flow = bool(dense_flow)
        self.reduce = _reduction(channels, bottleneck_channels)
        hidden_channels = max(8, bottleneck_channels)
        self.motion_estimator = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels * 2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.motion_estimator[-1].weight)
        nn.init.zeros_(self.motion_estimator[-1].bias)
        self.fusion = _fusion(bottleneck_channels * 4, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    @staticmethod
    def _base_grid(batch, height, width, device, dtype):
        y = (
            (torch.arange(height, device=device, dtype=dtype) + 0.5)
            * (2.0 / float(height)) - 1.0
        )
        x = (
            (torch.arange(width, device=device, dtype=dtype) + 0.5)
            * (2.0 / float(width)) - 1.0
        )
        grid_y = y[:, None].expand(height, width)
        grid_x = x[None, :].expand(height, width)
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(
            batch, -1, -1, -1
        )

    def _estimate_flow(self, appearance, reference):
        batch, channels, time, height, width = appearance.shape
        current_2d = appearance.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        reference_2d = reference.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        estimator_input = F.avg_pool2d(
            torch.cat([current_2d, reference_2d], dim=1),
            kernel_size=4,
            stride=4,
            ceil_mode=True,
        )
        flow_low = self.motion_estimator(estimator_input)
        if self.dense_flow:
            flow = F.interpolate(
                flow_low,
                size=(height, width),
                mode='bilinear',
                align_corners=False,
            )
        else:
            flow = flow_low.mean(dim=(2, 3), keepdim=True).expand(
                -1, -1, height, width
            )
        return torch.tanh(flow) * self.max_shift

    def _warp(self, appearance, flow):
        batch, channels, time, height, width = appearance.shape
        feature_2d = appearance.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        grid = self._base_grid(
            batch * time,
            height,
            width,
            feature_2d.device,
            feature_2d.dtype,
        ).clone()
        grid[..., 0] += flow[:, 0] * (2.0 / float(width))
        grid[..., 1] += flow[:, 1] * (2.0 / float(height))
        aligned = F.grid_sample(
            feature_2d,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
        return aligned.view(batch, time, channels, height, width).permute(
            0, 2, 1, 3, 4
        )

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        reference = appearance[:, :, appearance.shape[2] // 2:][:, :, :1]
        reference = reference.expand(-1, -1, appearance.shape[2], -1, -1)
        flow = self._estimate_flow(appearance, reference)
        aligned = self._warp(appearance, flow)
        anomaly = aligned - reference
        contrast = _local_contrast(aligned)
        fused = self.fusion(torch.cat([
            appearance, aligned, anomaly, contrast
        ], dim=1))
        reliability = self.gate(
            torch.cat([anomaly.abs(), contrast.abs()], dim=1)
        )
        delta = self.projection(fused)
        output = x + reliability * delta
        if not return_aux:
            return output
        batch, _, time, height, width = appearance.shape
        flow_5d = flow.view(batch, time, 2, height, width).permute(
            0, 2, 1, 3, 4
        )
        return output, {
            'flow': flow_5d,
            'aligned_anomaly': anomaly,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


class GlobalMotionAlignmentAdapter(_AlignmentAdapter):
    def __init__(self, channels=32, bottleneck_channels=8, max_shift=4.0):
        super().__init__(
            channels, bottleneck_channels, max_shift, dense_flow=False
        )


class LocalMotionAlignmentAdapter(_AlignmentAdapter):
    def __init__(self, channels=32, bottleneck_channels=8, max_shift=4.0):
        super().__init__(
            channels, bottleneck_channels, max_shift, dense_flow=True
        )


class MultiScaleContextAdapter(nn.Module):
    """Full-resolution multi-scale spatial head with residual fusion."""

    def __init__(self, channels=32, bottleneck_channels=8):
        super().__init__()
        self.reduce = _reduction(channels, bottleneck_channels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    bottleneck_channels,
                    bottleneck_channels,
                    kernel_size=(1, 3, 3),
                    padding=(0, dilation, dilation),
                    dilation=(1, dilation, dilation),
                    groups=bottleneck_channels,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(bottleneck_channels), bottleneck_channels
                ),
                nn.SiLU(inplace=True),
            )
            for dilation in (1, 2, 3)
        ])
        self.fusion = _fusion(bottleneck_channels * 5, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        branches = [branch(appearance) for branch in self.branches]
        contrast = _local_contrast(appearance)
        context = self.fusion(torch.cat(
            [appearance, contrast] + branches, dim=1
        ))
        reliability = self.gate(
            torch.cat([context.abs(), contrast.abs()], dim=1)
        )
        delta = self.projection(context)
        output = x + reliability * delta
        if not return_aux:
            return output
        return output, {
            'context': context,
            'local_contrast': contrast,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


class _DepthwiseConvGRUCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        joined_channels = channels * 2
        self.gates = nn.Sequential(
            nn.Conv2d(
                joined_channels, joined_channels, kernel_size=3, padding=1,
                groups=joined_channels, bias=False,
            ),
            nn.Conv2d(joined_channels, channels * 2, kernel_size=1),
        )
        self.candidate = nn.Sequential(
            nn.Conv2d(
                joined_channels, joined_channels, kernel_size=3, padding=1,
                groups=joined_channels, bias=False,
            ),
            nn.Conv2d(joined_channels, channels, kernel_size=1),
        )

    def forward(self, x, state):
        reset, update = torch.sigmoid(
            self.gates(torch.cat([x, state], dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * state], dim=1))
        )
        return (1.0 - update) * state + update * candidate


class BidirectionalPropagationAdapter(nn.Module):
    """Lightweight forward/backward recurrent propagation at full resolution."""

    def __init__(self, channels=32, bottleneck_channels=4):
        super().__init__()
        self.reduce = _reduction(channels, bottleneck_channels)
        self.forward_cell = _DepthwiseConvGRUCell(bottleneck_channels)
        self.backward_cell = _DepthwiseConvGRUCell(bottleneck_channels)
        self.fusion = _fusion(bottleneck_channels * 4, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def _propagate(self, appearance, reverse=False):
        indices = range(appearance.shape[2] - 1, -1, -1) if reverse else range(
            appearance.shape[2]
        )
        state = torch.zeros_like(appearance[:, :, 0])
        outputs = []
        cell = self.backward_cell if reverse else self.forward_cell
        for index in indices:
            state = cell(appearance[:, :, index], state)
            outputs.append(state)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=2)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        forward_features = self._propagate(appearance, reverse=False)
        backward_features = self._propagate(appearance, reverse=True)
        contrast = _local_contrast(appearance)
        fused = self.fusion(torch.cat([
            appearance, forward_features, backward_features, contrast
        ], dim=1))
        temporal_disagreement = (forward_features - backward_features).abs()
        reliability = self.gate(torch.cat([
            temporal_disagreement, contrast.abs()
        ], dim=1))
        delta = self.projection(fused)
        output = x + reliability * delta
        if not return_aux:
            return output
        return output, {
            'forward_features': forward_features,
            'backward_features': backward_features,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


class TDCDualStreamAdapter(nn.Module):
    """Parallel ordinary 3D context and explicit temporal-difference stream."""

    def __init__(self, channels=32, bottleneck_channels=8):
        super().__init__()
        self.reduce = _reduction(channels, bottleneck_channels)
        self.appearance_branches = nn.ModuleList()
        self.difference_branches = nn.ModuleList()
        for dilation in (1, 2, 4):
            branch = lambda: nn.Sequential(
                nn.Conv3d(
                    bottleneck_channels,
                    bottleneck_channels,
                    kernel_size=(3, 1, 1),
                    padding=(dilation, 0, 0),
                    dilation=(dilation, 1, 1),
                    groups=bottleneck_channels,
                    bias=False,
                ),
                nn.GroupNorm(
                    _group_count(bottleneck_channels), bottleneck_channels
                ),
                nn.SiLU(inplace=True),
            )
            self.appearance_branches.append(branch())
            self.difference_branches.append(branch())
        self.fusion = _fusion(bottleneck_channels * 4, bottleneck_channels)
        self.gate = _reliability_gate(bottleneck_channels)
        self.projection = _zero_projection(bottleneck_channels, channels)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        previous, following = _temporal_neighbours(appearance, 1)
        temporal_difference = 0.5 * (following - previous)
        appearance_context = sum(
            branch(appearance) for branch in self.appearance_branches
        ) / float(len(self.appearance_branches))
        difference_context = sum(
            branch(temporal_difference)
            for branch in self.difference_branches
        ) / float(len(self.difference_branches))
        contrast = _local_contrast(appearance)
        fused = self.fusion(torch.cat([
            appearance, appearance_context, difference_context, contrast
        ], dim=1))
        reliability = self.gate(torch.cat([
            difference_context.abs(), contrast.abs()
        ], dim=1))
        delta = self.projection(fused)
        output = x + reliability * delta
        if not return_aux:
            return output
        return output, {
            'appearance_context': appearance_context,
            'difference_context': difference_context,
            'reliability_gate': reliability,
            'residual_delta': delta,
        }


def build_structure_adapter(
    variant,
    channels,
    bottleneck_channels=8,
    max_shift=4.0,
):
    if variant not in STRUCTURE_VARIANTS:
        raise ValueError(
            'Unknown structure variant %s; expected one of %s'
            % (variant, ', '.join(STRUCTURE_VARIANTS))
        )
    if bottleneck_channels <= 0:
        raise ValueError('bottleneck_channels must be positive')
    if variant == 'second_order':
        return SecondOrderMotionAdapter(channels, bottleneck_channels)
    if variant in {'lfp_shallow', 'lfp_deep'}:
        return LowFrequencyPurificationAdapter(
            channels, bottleneck_channels
        )
    if variant == 'global_align':
        return GlobalMotionAlignmentAdapter(
            channels, bottleneck_channels, max_shift
        )
    if variant == 'local_align':
        return LocalMotionAlignmentAdapter(
            channels, bottleneck_channels, max_shift
        )
    if variant == 'multiscale_head':
        return MultiScaleContextAdapter(channels, bottleneck_channels)
    if variant == 'bidirectional':
        return BidirectionalPropagationAdapter(
            channels, min(4, bottleneck_channels)
        )
    if variant == 'tdc_dual_stream':
        return TDCDualStreamAdapter(channels, bottleneck_channels)
    raise AssertionError('unreachable structure variant')
