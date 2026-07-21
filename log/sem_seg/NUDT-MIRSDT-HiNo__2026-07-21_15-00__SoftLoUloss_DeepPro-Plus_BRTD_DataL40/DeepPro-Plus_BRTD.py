import torch
import torch.nn as nn

from networks.layers.basic import SDifferenceConv, STD_Resblock
from networks.layers.brtd_adapter import BRTDAdapter
from networks.layers.TPro import TPro
from networks.losses.HAM_loss_MultiFrame import HAM_loss
from networks.losses.HPM_loss_MultiFrame import HPM_loss


class detector(nn.Module):
    def __init__(
        self,
        num_classes,
        seqlen=100,
        out_len=100,
        use_background=True,
        adaptive_tdc=True,
        use_gate=True,
        zero_init=True,
        eval_chunk_rows=0,
    ):
        super().__init__()
        self.out_len = out_len
        self.eval_chunk_rows = int(eval_chunk_rows)
        if self.eval_chunk_rows < 0:
            raise ValueError('eval_chunk_rows must be non-negative.')

        # Keep the complete DeepPro-Plus stem and temporal-profile backbone.
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
        self.brtd = BRTDAdapter(
            channels=8,
            use_background=use_background,
            adaptive_tdc=adaptive_tdc,
            use_gate=use_gate,
            zero_init=zero_init,
        )
        self.layer1 = nn.Sequential(
            STD_Resblock(8, 16),
            STD_Resblock(16, 32),
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

        if return_aux:
            seq_feats, auxiliary = self.brtd(seq_feats, return_aux=True)
        else:
            seq_feats = self.brtd(seq_feats)

        seq_feats = self.layer1(seq_feats)
        seq_feats = seq_feats.permute(0, 3, 4, 1, 2)
        if (
            not self.training
            and self.eval_chunk_rows > 0
            and seq_feats.shape[1] > self.eval_chunk_rows
        ):
            decoded_chunks = []
            for row_start in range(
                0,
                seq_feats.shape[1],
                self.eval_chunk_rows,
            ):
                row_end = min(
                    row_start + self.eval_chunk_rows,
                    seq_feats.shape[1],
                )
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


class HAMloss(nn.Module):
    def __init__(
        self,
        alpha=(0.1667, 0.8333),
        gamma=2,
        MaxClutterNum=39,
        ProtectedArea=2,
    ):
        super().__init__()
        self.HAM = HAM_loss(
            alpha=list(alpha),
            gamma=gamma,
            MaxClutterNum=MaxClutterNum,
            ProtectedArea=ProtectedArea,
        )

    def forward(self, midpred, target):
        return self.HAM(midpred, target)


class HPMloss(nn.Module):
    def __init__(
        self,
        alpha=(0.1667, 0.8333),
        gamma=2,
        MaxClutterNum=39,
        ProtectedArea=2,
    ):
        super().__init__()
        self.HPM = HPM_loss(
            alpha=list(alpha),
            gamma=gamma,
            MaxClutterNum=MaxClutterNum,
            ProtectedArea=ProtectedArea,
        )

    def forward(self, midpred, target):
        return self.HPM(midpred, target)


class bceloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, midpred, target):
        return self.bce(midpred, target)


class SoftLoUloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, midpred, target):
        midpred = torch.sigmoid(midpred)
        intersection = midpred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(midpred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        iou = intersection_sum / (
            pred_sum + target_sum - intersection_sum
        )
        return 1 - torch.mean(iou)
