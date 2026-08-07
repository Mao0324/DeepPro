import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers.tdc import TDC


def _group_count(channels, max_groups=4):
    """Return a GroupNorm group count that divides ``channels``."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class RingBackgroundReference(nn.Module):
    """Build a local background-referenced residual at two spatial scales.

    The 3x3 ring describes the immediate neighbourhood and the 7x7-3x3
    ring describes a wider background.  A small per-location router chooses
    how much each reference contributes.
    """

    def __init__(self, channels):
        super().__init__()
        groups = _group_count(channels)
        self.router = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, 2, kernel_size=1, bias=True),
        )

    @staticmethod
    def _box_sum(x, kernel_size):
        pad = kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad, 0, 0), mode="replicate")
        mean = F.avg_pool3d(
            x,
            kernel_size=(1, kernel_size, kernel_size),
            stride=1,
            padding=0,
        )
        return mean * float(kernel_size * kernel_size)

    def forward(self, x):
        sum_3 = self._box_sum(x, 3)
        sum_7 = self._box_sum(x, 7)

        near_background = (sum_3 - x) / 8.0
        wide_background = (sum_7 - sum_3) / 40.0

        near_residual = x - near_background
        wide_residual = x - wide_background

        weights = torch.softmax(
            self.router(torch.cat([near_residual, wide_residual], dim=1)),
            dim=1,
        )
        residual = (
            weights[:, 0:1] * near_residual
            + weights[:, 1:2] * wide_residual
        )
        return residual, weights


class AdaptiveTDCR(nn.Module):
    """Depth-wise short/middle/long TDC branches with temporal routing."""

    def __init__(self, channels, adaptive=True):
        super().__init__()
        self.adaptive = adaptive

        def make_branch(step):
            return nn.Sequential(
                TDC(
                    channels,
                    channels,
                    kernel_size=(5, 1, 1),
                    stride=(1, 1, 1),
                    padding=(2, 0, 0),
                    groups=channels,
                    bias=False,
                    step=step,
                ),
                nn.BatchNorm3d(channels),
            )

        self.short_tdc = make_branch(step=1)
        self.middle_tdc = make_branch(step=2)
        self.long_tdc = make_branch(step=-1)

        # Mean and maximum energy from each of the three branches: 6 channels.
        self.temporal_router = nn.Sequential(
            nn.Conv1d(6, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(12),
            nn.SiLU(inplace=True),
            nn.Conv1d(12, 3, kernel_size=1, bias=True),
        )
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _branch_statistics(branches):
        mean_energy = torch.stack(
            [branch.abs().mean(dim=(1, 3, 4)) for branch in branches],
            dim=1,
        )
        max_energy = torch.stack(
            [branch.abs().amax(dim=(1, 3, 4)) for branch in branches],
            dim=1,
        )
        return torch.cat([mean_energy, max_energy], dim=1)

    def forward(self, x):
        branches = [
            self.short_tdc(x),
            self.middle_tdc(x),
            self.long_tdc(x),
        ]

        if self.adaptive:
            statistics = self._branch_statistics(branches)
            weights = torch.softmax(self.temporal_router(statistics), dim=1)
            weights = weights.unsqueeze(-1).unsqueeze(-1)
        else:
            weights = x.new_full(
                (x.shape[0], 3, x.shape[2], 1, 1),
                1.0 / 3.0,
            )

        fused = sum(
            weights[:, index:index + 1] * branch
            for index, branch in enumerate(branches)
        )
        return self.activation(fused), weights


class BRTDAdapter(nn.Module):
    """Background-referenced temporal-difference residual adapter.

    The projection is zero-initialized by default, so inserting this module
    into a pretrained DeepPro-Plus model initially leaves its prediction
    unchanged.  The new branch then learns only a residual correction.
    """

    def __init__(
        self,
        channels=8,
        use_background=True,
        adaptive_tdc=True,
        use_gate=True,
        zero_init=True,
    ):
        super().__init__()
        self.use_background = use_background
        self.use_gate = use_gate

        self.background_reference = RingBackgroundReference(channels)
        self.adaptive_tdcr = AdaptiveTDCR(channels, adaptive=adaptive_tdc)

        groups = _group_count(channels)
        self.reliability_gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.delta_projection = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        if zero_init:
            nn.init.zeros_(self.delta_projection.weight)

    def forward(self, x, return_aux=False):
        if self.use_background:
            reference_residual, reference_weights = self.background_reference(x)
        else:
            reference_residual = x
            reference_weights = None

        difference, temporal_weights = self.adaptive_tdcr(reference_residual)
        delta = self.delta_projection(difference)

        if self.use_gate:
            gate = self.reliability_gate(torch.cat([x, difference], dim=1))
        else:
            gate = torch.ones_like(delta)

        output = x + gate * delta

        if not return_aux:
            return output

        auxiliary = {
            "reference_weights": reference_weights,
            "temporal_weights": temporal_weights,
            "reliability_gate": gate,
            "residual_delta": delta,
        }
        return output, auxiliary
