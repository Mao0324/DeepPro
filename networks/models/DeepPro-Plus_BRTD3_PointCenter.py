"""Scratch Hybrid-RMS detector with restoration filtering and center peaks."""

import torch
import torch.nn as nn

from networks.layers.basic import SDifferenceConv, STD_Resblock
from networks.layers.TPro import TPro

try:
    # Experiment snapshots place the exact adapter source beside this model.
    from structure_adapters import build_structure_adapter
except ImportError:
    from networks.layers.structure_adapters import build_structure_adapter


class GatedRestorationBlock(nn.Module):
    """NAF-style local/temporal gated residual feature transformation."""

    def __init__(self, channels):
        super().__init__()
        expanded = channels * 2
        self.norm1 = nn.GroupNorm(1, channels)
        self.expand1 = nn.Conv3d(channels, expanded, 1)
        self.local_temporal = nn.Conv3d(
            expanded,
            expanded,
            kernel_size=3,
            padding=1,
            groups=expanded,
        )
        self.project1 = nn.Conv3d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.expand2 = nn.Conv3d(channels, expanded, 1)
        self.project2 = nn.Conv3d(channels, channels, 1)
        # Start as an identity mapping, as in NAF-style residual blocks.  The
        # scratch model otherwise compounds four multiplicative gates before
        # their scales have learned a safe magnitude under FP16.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    @staticmethod
    def _simple_gate(tensor):
        first, second = tensor.chunk(2, dim=1)
        return first * second

    def forward(self, features):
        transformed = self.local_temporal(self.expand1(self.norm1(features)))
        features = features + self.beta * self.project1(
            self._simple_gate(transformed)
        )
        transformed = self.expand2(self.norm2(features))
        return features + self.gamma * self.project2(
            self._simple_gate(transformed)
        )


class SpatialChannelModulator(nn.Module):
    """CBAM-style filtering that preserves the temporal dimension."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 2)
        self.channel_gate = nn.Sequential(
            nn.Conv3d(channels * 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(
                2,
                1,
                kernel_size=(1, 7, 7),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, features):
        channel_statistics = torch.cat([
            features.mean(dim=(-2, -1), keepdim=True),
            features.amax(dim=(-2, -1), keepdim=True),
        ], dim=1)
        modulated = features * self.channel_gate(channel_statistics)
        spatial_statistics = torch.cat([
            modulated.mean(dim=1, keepdim=True),
            modulated.amax(dim=1, keepdim=True),
        ], dim=1)
        return modulated * self.spatial_gate(spatial_statistics)


class ProgressiveCenterFilter(nn.Module):
    """Filter background redundancy and expose mask/center predictions."""

    def __init__(self, channels, num_classes, center_fusion_weight):
        super().__init__()
        self.center_fusion_weight = float(center_fusion_weight)
        if self.center_fusion_weight < 0.0:
            raise ValueError('center_fusion_weight must be non-negative.')
        self.pre_head = nn.Conv3d(channels, num_classes, 1)
        self.modulator = SpatialChannelModulator(channels)
        self.filter = nn.Sequential(
            GatedRestorationBlock(channels),
            GatedRestorationBlock(channels),
        )
        self.mask_head = nn.Conv3d(channels, num_classes, 1)
        self.center_head = nn.Conv3d(channels, num_classes, 1)
        # CenterNet's low-prior initialization prevents the dense full-resolution
        # focal head from overwhelming the overlap loss at the start of scratch
        # training (sigmoid(-2.19) is approximately 0.10).
        nn.init.constant_(self.center_head.bias, -2.19)

    def _forward_impl(self, features):
        pre_logits = self.pre_head(features)
        filter_input = features + self.modulator(features)
        filter_stage1 = self.filter[0](filter_input)
        # The next block no longer needs filter_input.  Releasing this full-
        # resolution tensor is essential for 1280x1024 test sequences and is
        # numerically identical to keeping the dead reference alive.
        del filter_input
        filtered = self.filter[1](filter_stage1)
        del filter_stage1
        mask_logits = self.mask_head(filtered)
        center_logits = self.center_head(filtered)
        fused_logits = (
            mask_logits + self.center_fusion_weight * center_logits
        )
        auxiliary = {
            'pre_logits': pre_logits.squeeze(1),
            'center_logits': center_logits.squeeze(1),
            'mask_logits': mask_logits.squeeze(1),
        }
        return auxiliary, fused_logits.squeeze(1)

    def forward(self, features):
        if not features.is_cuda:
            return self._forward_impl(features)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                'PointCenter filtering requires CUDA BF16 support.'
            )
        # The second multiplicative gate can exceed FP16's exponent range for
        # a single hard clip.  BF16 retains two-byte activations while avoiding
        # that overflow; the upstream backbone remains under FP16 autocast.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return self._forward_impl(features.to(dtype=torch.bfloat16))


class detector(nn.Module):
    def __init__(
        self,
        num_classes,
        seqlen=40,
        out_len=40,
        structure_variant='raw_apmd_hybrid_rms',
        structure_bottleneck_channels=8,
        structure_max_shift=4.0,
        eval_chunk_rows=0,
        point_center_fusion_weight=0.25,
    ):
        super().__init__()
        self.out_len = int(out_len)
        self.structure_variant = structure_variant
        self.eval_chunk_rows = int(eval_chunk_rows)
        if self.eval_chunk_rows < 0:
            raise ValueError('eval_chunk_rows must be non-negative')
        if not structure_variant.startswith('raw_apmd'):
            raise ValueError(
                'PointCenter requires a raw_apmd structural adapter, got %r.'
                % structure_variant
            )

        self.conv_in = nn.Sequential(
            SDifferenceConv(
                in_channels=1,
                out_channels=8,
                kernel_size=(5, 7, 7),
                stride=(1, 1, 1),
                padding=(2, 3, 3),
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            STD_Resblock(8, 16),
            STD_Resblock(16, 32),
        )
        self.brtd = build_structure_adapter(
            structure_variant,
            channels=32,
            bottleneck_channels=structure_bottleneck_channels,
            max_shift=structure_max_shift,
        )
        self.brtd.low_memory_eval = self.eval_chunk_rows > 0
        self.TPro = TPro(
            d_model=32,
            num_head=8,
            seqlen=seqlen,
            out_len=out_len,
        )
        self.feature_projection = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.restoration_blocks = nn.Sequential(
            GatedRestorationBlock(16),
            GatedRestorationBlock(16),
        )
        self.output_filter = ProgressiveCenterFilter(
            16,
            num_classes,
            center_fusion_weight=point_center_fusion_weight,
        )

    def _temporal_projection(self, features):
        features = features.permute(0, 3, 4, 1, 2)
        if (
            not self.training
            and self.eval_chunk_rows > 0
            and features.shape[1] > self.eval_chunk_rows
        ):
            projected_chunks = []
            for row_start in range(0, features.shape[1], self.eval_chunk_rows):
                row_end = min(
                    row_start + self.eval_chunk_rows, features.shape[1]
                )
                projected_chunks.append(
                    self.feature_projection(
                        self.TPro(features[:, row_start:row_end])
                    )
                )
            return torch.cat(projected_chunks, dim=3)
        return self.feature_projection(self.TPro(features))

    def forward(self, seq_imgs):
        if seq_imgs.ndim != 5 or seq_imgs.shape[1] != 1:
            raise ValueError('expected input shape [B,1,T,H,W]')
        if seq_imgs.shape[2] != self.out_len:
            raise ValueError(
                'expected %d frames, got %d'
                % (self.out_len, seq_imgs.shape[2])
            )
        features = self.layer1(self.conv_in(seq_imgs))
        features = self.brtd(features, seq_imgs)
        features = self._temporal_projection(features)
        if features.is_cuda:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    'PointCenter restoration requires CUDA BF16 support.'
                )
            # Keep every multiplicative restoration gate out of FP16.  Patch
            # training can remain finite while a rare full-resolution value
            # overflows before the small residual scale is applied.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                features = self.restoration_blocks(
                    features.to(dtype=torch.bfloat16)
                )
        else:
            features = self.restoration_blocks(features)
        return self.output_filter(features)
