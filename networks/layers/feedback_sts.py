"""Scratch-only spatio-temporal semantic feedback building blocks.

The implementation follows the architectural ideas of FeedbackSTS-Det while
using torchvision's maintained deformable convolution operator.  It does not
load or depend on any pretrained parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


def _offset_groups(channels, maximum=4):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class PyramidDeformableAlignment(nn.Module):
    """Align a frame to a reference through a coarse-to-fine pyramid."""

    def __init__(self, channels, levels=2, maximum_offset_groups=4):
        super().__init__()
        if levels < 2:
            raise ValueError('alignment levels must be at least two')
        self.levels = int(levels)
        self.offset_groups = _offset_groups(
            channels, maximum=maximum_offset_groups
        )
        offset_mask_channels = 3 * 3 * 3 * self.offset_groups

        self.downsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.levels - 1)
        ])
        self.offset_conv1 = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, 3, padding=1)
            for _ in range(self.levels)
        ])
        self.offset_conv2 = nn.ModuleList([
            nn.Conv2d(
                channels if level == self.levels - 1 else channels * 2,
                channels,
                3,
                padding=1,
            )
            for level in range(self.levels)
        ])
        self.offset_conv3 = nn.ModuleList([
            nn.Identity() if level == self.levels - 1 else nn.Conv2d(
                channels, channels, 3, padding=1
            )
            for level in range(self.levels)
        ])
        self.offset_mask = nn.ModuleList([
            nn.Conv2d(channels, offset_mask_channels, 3, padding=1)
            for _ in range(self.levels)
        ])
        self.deform = nn.ModuleList([
            DeformConv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            for _ in range(self.levels)
        ])
        self.fuse = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, 3, padding=1)
            for _ in range(self.levels - 1)
        ])

        # Match DCNv2's stable start: zero displacement and a 0.5 modulation
        # mask. The offset-feature path starts learning after this projection's
        # first update, while the rest of the detector receives gradients from
        # the first step.
        for predictor in self.offset_mask:
            nn.init.zeros_(predictor.weight)
            nn.init.zeros_(predictor.bias)

    def _pyramid(self, tensor):
        features = [tensor]
        for block in self.downsample:
            features.append(block(features[-1]))
        return features

    def forward(self, neighbor, reference):
        if neighbor.shape != reference.shape:
            raise ValueError(
                'alignment tensors must match, got %s and %s'
                % (tuple(neighbor.shape), tuple(reference.shape))
            )
        neighbor_levels = self._pyramid(neighbor)
        reference_levels = self._pyramid(reference)
        coarse_offset_feature = None
        coarse_feature = None

        for level in range(self.levels - 1, -1, -1):
            offset_feature = F.relu(self.offset_conv1[level](torch.cat([
                neighbor_levels[level], reference_levels[level]
            ], dim=1)), inplace=True)
            if coarse_offset_feature is None:
                offset_feature = F.relu(
                    self.offset_conv2[level](offset_feature), inplace=True
                )
            else:
                offset_feature = F.relu(self.offset_conv2[level](torch.cat([
                    offset_feature,
                    F.interpolate(
                        coarse_offset_feature,
                        size=offset_feature.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    ) * 2.0,
                ], dim=1)), inplace=True)
                offset_feature = F.relu(
                    self.offset_conv3[level](offset_feature), inplace=True
                )
            offset_x, offset_y, modulation = torch.chunk(
                self.offset_mask[level](offset_feature), 3, dim=1
            )
            offset = torch.cat([offset_x, offset_y], dim=1)
            aligned = self.deform[level](
                neighbor_levels[level].contiguous(),
                offset.contiguous(),
                torch.sigmoid(modulation).contiguous(),
            )
            if coarse_feature is not None:
                aligned = self.fuse[level](torch.cat([
                    aligned,
                    F.interpolate(
                        coarse_feature,
                        size=aligned.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    ),
                ], dim=1))
            if level > 0:
                coarse_offset_feature = offset_feature
                coarse_feature = F.relu(aligned, inplace=False)
        return aligned


class SparseSemanticPropagation(nn.Module):
    """Propagate aligned semantics in fixed-interval temporal groups."""

    def __init__(
        self,
        channels,
        interval=2,
        forward=True,
        alignment_levels=2,
    ):
        super().__init__()
        if interval < 1:
            raise ValueError('temporal interval must be positive')
        self.interval = int(interval)
        self.forward_direction = bool(forward)
        self.alignment = PyramidDeformableAlignment(
            channels, levels=alignment_levels
        )

    def _forward_serial(self, sequence):
        """Reference implementation for unequal temporal groups."""
        frame_count = sequence.shape[2]
        outputs = [None] * frame_count
        for group_start in range(min(self.interval, frame_count)):
            indices = list(range(
                group_start, frame_count, self.interval
            ))
            if not self.forward_direction:
                indices.reverse()
            propagated = sequence[:, :, indices[0]]
            outputs[indices[0]] = propagated
            for frame_index in indices[1:]:
                current = sequence[:, :, frame_index]
                propagated = self.alignment(current, propagated)
                outputs[frame_index] = propagated
        return torch.stack(outputs, dim=2)

    def _forward_equal_groups(self, sequence):
        """Evaluate independent temporal residue chains as one batch."""
        batch, channels, frame_count, height, width = sequence.shape
        step_count = frame_count // self.interval
        groups = torch.stack([
            sequence[:, :, group_start::self.interval]
            for group_start in range(self.interval)
        ], dim=1).reshape(
            batch * self.interval,
            channels,
            step_count,
            height,
            width,
        )
        if not self.forward_direction:
            groups = groups.flip(2)

        propagated = groups[:, :, 0]
        outputs = [propagated]
        for step_index in range(1, step_count):
            propagated = self.alignment(
                groups[:, :, step_index], propagated
            )
            outputs.append(propagated)
        outputs = torch.stack(outputs, dim=2)
        if not self.forward_direction:
            outputs = outputs.flip(2)

        # [B, interval, C, steps, H, W] -> original interleaved frames.
        return outputs.reshape(
            batch,
            self.interval,
            channels,
            step_count,
            height,
            width,
        ).permute(0, 2, 3, 1, 4, 5).reshape(
            batch, channels, frame_count, height, width
        )

    def forward(self, sequence):
        if sequence.ndim != 5:
            raise ValueError('expected [B,C,T,H,W] sequence')
        frame_count = sequence.shape[2]
        if frame_count < 2:
            return sequence
        if (
            self.interval > 1
            and frame_count >= self.interval
            and frame_count % self.interval == 0
        ):
            return self._forward_equal_groups(sequence)
        return self._forward_serial(sequence)


class SpatioTemporalFeedbackBlock(nn.Module):
    """3D context path plus forward or backward semantic propagation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        interval=2,
        forward=True,
        alignment_levels=2,
    ):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.propagation = SparseSemanticPropagation(
            out_channels,
            interval=interval,
            forward=forward,
            alignment_levels=alignment_levels,
        )

    def forward(self, tensor):
        return self.context(tensor) + self.propagation(
            self.residual(tensor)
        )


class FeedbackSTSBackbone(nn.Module):
    """Bidirectional 3D U-Net with sparse semantic feedback at every scale."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64, 128),
        interval=2,
        alignment_levels=2,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError('at least two encoder levels are required')
        self.channels = tuple(int(value) for value in channels)
        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        previous = in_channels
        for channel_count in self.channels:
            self.encoder.append(SpatioTemporalFeedbackBlock(
                previous,
                channel_count,
                interval=interval,
                forward=True,
                alignment_levels=alignment_levels,
            ))
            self.pool.append(nn.MaxPool3d((1, 2, 2)))
            previous = channel_count

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for deep, shallow in zip(
            reversed(self.channels[1:]), reversed(self.channels[:-1])
        ):
            self.up.append(nn.ConvTranspose3d(
                deep,
                shallow,
                kernel_size=(1, 4, 4),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ))
            self.decoder.append(SpatioTemporalFeedbackBlock(
                shallow * 2,
                shallow,
                interval=interval,
                forward=False,
                alignment_levels=alignment_levels,
            ))
        self.head = nn.Conv3d(self.channels[0], out_channels, 1)

    @property
    def spatial_divisor(self):
        return 2 ** (len(self.channels) - 1)

    def forward(self, sequence):
        height, width = sequence.shape[-2:]
        divisor = self.spatial_divisor
        if height % divisor or width % divisor:
            raise ValueError(
                'spatial dimensions must be divisible by %d, got %dx%d'
                % (divisor, height, width)
            )
        skip_features = []
        current = sequence
        for level, block in enumerate(self.encoder):
            current = block(current)
            skip_features.append(current)
            if level + 1 < len(self.encoder):
                current = self.pool[level](current)

        for index, (upsample, block) in enumerate(zip(
            self.up, self.decoder
        )):
            current = F.relu(upsample(current), inplace=True)
            skip = skip_features[-2 - index]
            current = block(torch.cat([current, skip], dim=1))
        return current, self.head(current)


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())
