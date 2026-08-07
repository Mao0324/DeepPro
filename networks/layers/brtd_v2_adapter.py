"""Stable target-preserving BRTD adapter for infrared video features.

The adapter keeps an explicit appearance path, models several actual temporal
ranges with dilated depth-wise convolutions, and only applies the learned
correction through a conservative reliability gate. Its output projection can
be zero-initialized, making it safe to insert into a pretrained DeepPro-Plus
backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels, max_groups=4):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DilatedTemporalBranch(nn.Module):
    """Depth-wise temporal filtering without forcing a high-pass response."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.filter = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(dilation, 0, 0),
            dilation=(dilation, 1, 1),
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.filter(x)))


class StableBRTDAdapter(nn.Module):
    """Target-preserving background-referenced temporal adapter.

    Compared with the first BRTD adapter, this version preserves an appearance
    path, uses dilations for genuinely different temporal receptive fields,
    avoids running-stat normalization in the router, and uses local contrast
    as reliability evidence rather than as a destructive replacement input.
    """

    def __init__(
        self,
        channels=32,
        bottleneck_channels=8,
        temporal_dilations=(1, 2, 4),
        use_background=True,
        adaptive_tdc=True,
        use_gate=True,
        zero_init=True,
        gate_bias=-2.0,
    ):
        super().__init__()
        if channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("channels and bottleneck_channels must be positive")
        temporal_dilations = tuple(int(value) for value in temporal_dilations)
        if not temporal_dilations or any(value <= 0 for value in temporal_dilations):
            raise ValueError("temporal_dilations must contain positive integers")

        self.use_background = bool(use_background)
        self.adaptive_tdc = bool(adaptive_tdc)
        self.use_gate = bool(use_gate)
        self.temporal_dilations = temporal_dilations

        groups = _group_count(bottleneck_channels)
        self.reduce = nn.Sequential(
            nn.Conv3d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, bottleneck_channels),
            nn.SiLU(inplace=True),
        )
        self.temporal_branches = nn.ModuleList([
            DilatedTemporalBranch(bottleneck_channels, dilation)
            for dilation in temporal_dilations
        ])

        branch_count = len(temporal_dilations)
        router_hidden = max(8, branch_count * 4)
        self.temporal_router = nn.Sequential(
            nn.Conv1d(branch_count * 2, router_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(router_hidden, branch_count, kernel_size=1),
        )
        # Equal branch weights at initialization are more stable than a random
        # preference for one temporal scale.
        nn.init.zeros_(self.temporal_router[-1].weight)
        nn.init.zeros_(self.temporal_router[-1].bias)

        fusion_inputs = bottleneck_channels * (3 if self.use_background else 2)
        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_inputs, bottleneck_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, bottleneck_channels),
            nn.SiLU(inplace=True),
        )

        self.reliability_gate = nn.Sequential(
            nn.Conv3d(bottleneck_channels * 2, bottleneck_channels, kernel_size=1),
            nn.GroupNorm(groups, bottleneck_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(bottleneck_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.reliability_gate[-2].weight)
        nn.init.constant_(self.reliability_gate[-2].bias, float(gate_bias))

        self.delta_projection = nn.Conv3d(
            bottleneck_channels, channels, kernel_size=1, bias=False
        )
        if zero_init:
            nn.init.zeros_(self.delta_projection.weight)

    @staticmethod
    def _local_contrast(x):
        near_background = F.avg_pool3d(
            x, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )
        wide_background = F.avg_pool3d(
            x, kernel_size=(1, 7, 7), stride=1, padding=(0, 3, 3)
        )
        return x - 0.5 * (near_background + wide_background)

    @staticmethod
    def _branch_statistics(branches):
        mean_energy = torch.stack(
            [branch.abs().mean(dim=(1, 3, 4)) for branch in branches], dim=1
        )
        max_energy = torch.stack(
            [branch.abs().amax(dim=(1, 3, 4)) for branch in branches], dim=1
        )
        return torch.cat([mean_energy, max_energy], dim=1)

    def forward(self, x, return_aux=False):
        appearance = self.reduce(x)
        branches = [branch(appearance) for branch in self.temporal_branches]

        if self.adaptive_tdc:
            statistics = self._branch_statistics(branches)
            temporal_weights = torch.softmax(self.temporal_router(statistics), dim=1)
            weights_5d = temporal_weights.unsqueeze(-1).unsqueeze(-1)
        else:
            branch_count = len(branches)
            temporal_weights = x.new_full(
                (x.shape[0], branch_count, x.shape[2]), 1.0 / branch_count
            )
            weights_5d = temporal_weights.unsqueeze(-1).unsqueeze(-1)

        temporal_context = sum(
            weights_5d[:, index:index + 1] * branch
            for index, branch in enumerate(branches)
        )
        local_contrast = self._local_contrast(appearance)

        fusion_parts = [appearance, temporal_context]
        if self.use_background:
            fusion_parts.append(local_contrast)
        fused = self.fusion(torch.cat(fusion_parts, dim=1))

        if self.use_gate:
            gate_evidence = torch.cat([
                (temporal_context - appearance).abs(),
                local_contrast.abs(),
            ], dim=1)
            reliability = self.reliability_gate(gate_evidence)
        else:
            reliability = x.new_ones((x.shape[0], 1, *x.shape[2:]))

        residual_delta = self.delta_projection(fused)
        output = x + reliability * residual_delta

        if not return_aux:
            return output
        auxiliary = {
            "temporal_weights": temporal_weights,
            "reliability_gate": reliability,
            "residual_delta": residual_delta,
            "local_contrast": local_contrast,
        }
        return output, auxiliary
