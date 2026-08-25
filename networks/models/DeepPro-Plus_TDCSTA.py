import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import TD_Resblock, STD_Resblock
from networks.layers.basic import TDifferenceConv, SDifferenceConv
from networks.layers.TPro import TPro
from networks.layers.tdc import TDCR
from networks.layers.tdcsta import SelfAttention, CrossAttention
from networks.losses.HAM_loss_MultiFrame import HAM_loss
from networks.losses.HPM_loss_MultiFrame import HPM_loss
import numpy as np

class TDCSTAFront(nn.Module):
    def __init__(self, dim=32):
        super(TDCSTAFront, self).__init__()

        self.tdc_branch = TDCR(
            in_channels=1,
            out_channels=dim,
            kernel_size=(5, 3, 3),
            stride=(1, 1, 1),
            padding=(2, 1, 1)
        )

        self.st_branch = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(5, 7, 7), stride=(1, 1, 1), padding=(2, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            STD_Resblock(8, 16),
            STD_Resblock(16, dim)
        )

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.q_sa = SelfAttention(dim, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.k_sa = SelfAttention(dim, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.v_sa = SelfAttention(dim, window_size=(2, 8, 8), num_heads=4, use_shift=True, mlp_ratio=1.5)
        self.ca = CrossAttention(dim, window_size=(2, 8, 8), num_heads=4)

        self.freeze_st = False
        self.freeze_spatial = False 

    def load_pretrained_branches(self, spatial_ckpt=None, st_ckpt=None):
        if spatial_ckpt or st_ckpt:
            raise ValueError(
                "Scratch-only policy forbids loading spatial/st branch checkpoints"
            )
        if spatial_ckpt is not None and spatial_ckpt != "":
            state = torch.load(
                spatial_ckpt,
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(state, dict) and "branch_state_dict" in state:
                if state.get("stage") not in (None, "2d"):
                    raise ValueError(f"Expected a 2d spatial checkpoint, but got stage={state.get('stage')}")
                state = state["branch_state_dict"]
            self.spatial_branch.load_state_dict(state, strict=True)
            print(f"Loaded spatial_branch from {spatial_ckpt}")

        if st_ckpt is not None and st_ckpt != "":
            state = torch.load(
                st_ckpt,
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(state, dict) and "branch_state_dict" in state:
                if state.get("stage") not in (None, "3d"):
                    raise ValueError(f"Expected a 3d spatio-temporal checkpoint, but got stage={state.get('stage')}")
                state = state["branch_state_dict"]
            self.st_branch.load_state_dict(state, strict=True)
            print(f"Loaded st_branch from {st_ckpt}")


    def freeze_pretrained_backbones(self, freeze_spatial=True, freeze_st=True):
        self.freeze_spatial = freeze_spatial
        self.freeze_st = freeze_st

        if freeze_st:
            self.st_branch.eval()
            for p in self.st_branch.parameters():
                p.requires_grad = False

        if freeze_spatial:
            self.spatial_branch.eval()
            for p in self.spatial_branch.parameters():
                p.requires_grad = False


    def forward(self, seq_imgs):
        q = self.tdc_branch(seq_imgs)

        if self.freeze_st:
            self.st_branch.eval()
            with torch.no_grad():
                k = self.st_branch(seq_imgs)
        else:
            k = self.st_branch(seq_imgs)

        current = seq_imgs[:, :, -1, :, :]

        if self.freeze_spatial:
            self.spatial_branch.eval()
            with torch.no_grad():
                v = self.spatial_branch(current)
        else:
            v = self.spatial_branch(current)

        v = v.unsqueeze(2).expand(-1, -1, q.shape[2], -1, -1)

        q = q.permute(0, 2, 3, 4, 1)
        k = k.permute(0, 2, 3, 4, 1)
        v = v.permute(0, 2, 3, 4, 1)

        q = self.q_sa(q)
        k = self.k_sa(k)
        v = self.v_sa(v)

        out = self.ca(q, k, v)
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        return out

class detector(nn.Module):
    def __init__(self, num_classes, seqlen=100, out_len=100,
                 spatial_ckpt=None, st_ckpt=None, freeze_pretrained=False):
        super(detector, self).__init__()
        self.out_len = out_len
        # self.conv_in = nn.Sequential(SDifferenceConv(in_channels=1, out_channels=8, kernel_size=(5,7,7), stride=(1,1,1), padding=(2,3,3)),
        #                              nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        # self.layer1 = nn.Sequential(STD_Resblock(8, 16), STD_Resblock(16, 32))
        self.front = TDCSTAFront(dim=32)
        self.front.load_pretrained_branches(
            spatial_ckpt=spatial_ckpt,
            st_ckpt=st_ckpt
        )

        if freeze_pretrained:
            self.front.freeze_pretrained_backbones(
                freeze_spatial=spatial_ckpt is not None and spatial_ckpt != "",
                freeze_st=st_ckpt is not None and st_ckpt != ""
            )

        self.TPro = TPro(d_model=32, num_head=8, seqlen=seqlen, out_len=out_len)
        self.conv_out1 = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0)),
                                       nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.conv_out2 = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))


    def forward(self, seq_imgs):  ## 1.415G
        # seq_feats = self.conv_in(seq_imgs)  ## 3.171G     [:, :, :29, :, :]
        # seq_feats = self.layer1(seq_feats)  ## 20.771G

        seq_feats = self.front(seq_imgs)

        seq_feats = seq_feats.permute(0, 3, 4, 1, 2)
        seq_feats = self.TPro(seq_feats)

        seq_feats = self.conv_out1(seq_feats)
        seq_midout = self.conv_out2(seq_feats)
        seq_midseg = seq_midout.squeeze(dim=1)    ## b, t, h, w

        return seq_feats, seq_midseg



class HAMloss(nn.Module):
    def __init__(self, alpha=[0.1667, 0.8333], gamma=2, MaxClutterNum=39, ProtectedArea=2):
        super(HAMloss, self).__init__()
        self.HAM = HAM_loss(alpha=alpha, gamma=gamma, MaxClutterNum=MaxClutterNum, ProtectedArea=ProtectedArea)

    def forward(self, midpred, target):

        b, t, h, w = midpred.size()
        # input = midpred.view(b*t, h, w).unsqueeze(dim=1)
        # target = target.view(b*t, h, w).unsqueeze(dim=1)
        loss_mid = self.HAM(midpred, target)

        return loss_mid



class HPMloss(nn.Module):
    def __init__(self, alpha=[0.1667, 0.8333], gamma=2, MaxClutterNum=39, ProtectedArea=2):
        super(HPMloss, self).__init__()
        self.HPM = HPM_loss(alpha=alpha, gamma=gamma, MaxClutterNum=MaxClutterNum, ProtectedArea=ProtectedArea)

    def forward(self, midpred, target):

        b, t, h, w = midpred.size()
        # input = midpred.view(b*t, h, w).unsqueeze(dim=1)
        # target = target.view(b*t, h, w).unsqueeze(dim=1)
        loss_mid = self.HPM(midpred, target)

        return loss_mid



class bceloss(nn.Module):
    def __init__(self):
        super(bceloss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, midpred, target):
        loss_mid = self.bce(midpred, target)

        return loss_mid



class SoftLoUloss(nn.Module):
    def __init__(self):
        super(SoftLoUloss, self).__init__()

    def forward(self, midpred, target):
        smooth = 0.00
        midpred = torch.sigmoid(midpred)
        intersection = midpred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(midpred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        union = pred_sum + target_sum - intersection_sum
        loss_mid = torch.where(
            union > 0,
            intersection_sum / union.clamp_min(torch.finfo(union.dtype).tiny),
            torch.ones_like(union),
        )

        loss_mid = 1 - torch.mean(loss_mid)

        return loss_mid


# if __name__ == '__main__':
#     import  torch
#     model = generator(1)
#     seq_imgs = torch.rand(1, 100, 512, 512)
#     (model(seq_imgs))
