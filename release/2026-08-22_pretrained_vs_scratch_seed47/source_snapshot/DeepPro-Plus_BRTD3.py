"""DeepPro-Plus with selectable zero-initialized structural adapters."""

import torch
import torch.nn as nn

from networks.layers.basic import SDifferenceConv, STD_Resblock
from networks.layers.TPro import TPro

try:
    # Experiment snapshots place the exact adapter source beside this model.
    from structure_adapters import build_structure_adapter
except ImportError:
    from networks.layers.structure_adapters import build_structure_adapter


class detector(nn.Module):
    def __init__(
        self,
        num_classes,
        seqlen=100,
        out_len=100,
        structure_variant='second_order',
        structure_bottleneck_channels=8,
        structure_max_shift=4.0,
        eval_chunk_rows=0,
    ):
        super().__init__()
        self.out_len = out_len
        self.structure_variant = structure_variant
        self.eval_chunk_rows = int(eval_chunk_rows)
        if self.eval_chunk_rows < 0:
            raise ValueError('eval_chunk_rows must be non-negative')

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

        if structure_variant == 'lfp_shallow':
            adapter_channels = 8
            self.adapter_position = 'shallow'
        elif structure_variant.startswith('raw_apmd'):
            adapter_channels = 32
            self.adapter_position = 'raw_fusion'
        elif structure_variant == 'multiscale_head':
            adapter_channels = 32
            self.adapter_position = 'post_tpro'
        else:
            adapter_channels = 32
            self.adapter_position = 'deep'
        self.brtd = build_structure_adapter(
            structure_variant,
            channels=adapter_channels,
            bottleneck_channels=structure_bottleneck_channels,
            max_shift=structure_max_shift,
        )

        self.TPro = TPro(
            d_model=32,
            num_head=8,
            seqlen=seqlen,
            out_len=out_len,
        )
        self.conv_out1 = nn.Sequential(
            nn.Conv3d(32, 8, kernel_size=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv_out2 = nn.Conv3d(8, num_classes, kernel_size=1)

    def _apply_adapter(
        self, features, return_aux, raw_frames=None
    ):
        if self.adapter_position == 'raw_fusion':
            if raw_frames is None:
                raise ValueError('raw_fusion adapter requires raw_frames')
            if return_aux:
                return self.brtd(
                    features, raw_frames, return_aux=True
                )
            return self.brtd(features, raw_frames), None
        if return_aux:
            return self.brtd(features, return_aux=True)
        return self.brtd(features), None

    def forward(self, seq_imgs, return_aux=False):
        seq_feats = self.conv_in(seq_imgs)
        auxiliary = None
        if self.adapter_position == 'shallow':
            seq_feats, auxiliary = self._apply_adapter(
                seq_feats, return_aux
            )
        seq_feats = self.layer1(seq_feats)
        if self.adapter_position in {'deep', 'raw_fusion'}:
            seq_feats, auxiliary = self._apply_adapter(
                seq_feats, return_aux, raw_frames=seq_imgs
            )

        seq_feats = seq_feats.permute(0, 3, 4, 1, 2)
        if (
            not self.training
            and self.eval_chunk_rows > 0
            and seq_feats.shape[1] > self.eval_chunk_rows
        ):
            temporal_chunks = []
            for row_start in range(
                0, seq_feats.shape[1], self.eval_chunk_rows
            ):
                row_end = min(
                    row_start + self.eval_chunk_rows,
                    seq_feats.shape[1],
                )
                temporal_chunks.append(
                    self.TPro(seq_feats[:, row_start:row_end])
                )
            if self.adapter_position == 'post_tpro':
                # TPro itself has no spatial mixing, so row chunks can be
                # reassembled before the spatial head.  This avoids artificial
                # seams at dilation radii 1/2/3 while retaining chunked TPro.
                seq_feats = torch.cat(temporal_chunks, dim=3)
                seq_feats, auxiliary = self._apply_adapter(
                    seq_feats, return_aux
                )
                seq_feats = self.conv_out1(seq_feats)
            else:
                seq_feats = torch.cat([
                    self.conv_out1(chunk) for chunk in temporal_chunks
                ], dim=3)
        else:
            seq_feats = self.TPro(seq_feats)
            if self.adapter_position == 'post_tpro':
                seq_feats, auxiliary = self._apply_adapter(
                    seq_feats, return_aux
                )
            seq_feats = self.conv_out1(seq_feats)

        seq_midseg = self.conv_out2(seq_feats).squeeze(dim=1)
        if return_aux:
            return seq_feats, seq_midseg, auxiliary
        return seq_feats, seq_midseg
