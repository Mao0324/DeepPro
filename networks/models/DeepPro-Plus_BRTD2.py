import torch
import torch.nn as nn

from networks.layers.basic import SDifferenceConv, STD_Resblock
from networks.layers.TPro import TPro

try:
    # Experiment snapshots place the adapter beside this model file.
    from brtd_v2_adapter import StableBRTDAdapter
except ImportError:
    from networks.layers.brtd_v2_adapter import StableBRTDAdapter


class detector(nn.Module):
    """DeepPro-Plus with a stable, target-preserving BRTD2 adapter."""

    def __init__(
        self,
        num_classes,
        seqlen=100,
        out_len=100,
        use_background=True,
        adaptive_tdc=True,
        use_gate=True,
        zero_init=True,
        bottleneck_channels=8,
        temporal_dilations=(1, 2, 4),
        gate_bias=-2.0,
        eval_chunk_rows=0,
    ):
        super().__init__()
        self.out_len = out_len
        self.eval_chunk_rows = int(eval_chunk_rows)
        if self.eval_chunk_rows < 0:
            raise ValueError("eval_chunk_rows must be non-negative")

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
        # Refine deeper features so weak target evidence has acquired semantic
        # support before temporal/background modulation.
        self.brtd = StableBRTDAdapter(
            channels=32,
            bottleneck_channels=bottleneck_channels,
            temporal_dilations=temporal_dilations,
            use_background=use_background,
            adaptive_tdc=adaptive_tdc,
            use_gate=use_gate,
            zero_init=zero_init,
            gate_bias=gate_bias,
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

    def forward(self, seq_imgs, return_aux=False):
        seq_feats = self.conv_in(seq_imgs)
        seq_feats = self.layer1(seq_feats)
        if return_aux:
            seq_feats, auxiliary = self.brtd(seq_feats, return_aux=True)
        else:
            seq_feats = self.brtd(seq_feats)

        seq_feats = seq_feats.permute(0, 3, 4, 1, 2)
        if (
            not self.training
            and self.eval_chunk_rows > 0
            and seq_feats.shape[1] > self.eval_chunk_rows
        ):
            decoded_chunks = []
            for row_start in range(0, seq_feats.shape[1], self.eval_chunk_rows):
                row_end = min(row_start + self.eval_chunk_rows, seq_feats.shape[1])
                chunk = self.TPro(seq_feats[:, row_start:row_end])
                decoded_chunks.append(self.conv_out1(chunk))
            seq_feats = torch.cat(decoded_chunks, dim=3)
        else:
            seq_feats = self.TPro(seq_feats)
            seq_feats = self.conv_out1(seq_feats)

        seq_midseg = self.conv_out2(seq_feats).squeeze(dim=1)
        if return_aux:
            return seq_feats, seq_midseg, auxiliary
        return seq_feats, seq_midseg
