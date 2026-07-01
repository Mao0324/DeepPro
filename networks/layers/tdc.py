import torch
import torch.nn as nn
import torch.nn.functional as F


class TDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 1, 1),
                 stride=(1, 1, 1), padding=(2, 0, 0), groups=1,
                 bias=False, step=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.step = step
        self.groups = groups

    def get_time_gradient_weight(self):
        weight = self.conv.weight
        kT = weight.shape[2]
        grad_weight = torch.zeros_like(weight)

        if kT == 5:
            if self.step == -1:
                grad_weight[:] = -weight
                grad_weight[:, :, 4] = weight[:, :, 0] + weight[:, :, 1] + weight[:, :, 2] + weight[:, :, 3] + weight[:, :, 4]
            elif self.step == 1:
                grad_weight[:, :, 4] = weight[:, :, 4]
                grad_weight[:, :, 3] = weight[:, :, 3] - weight[:, :, 4]
                grad_weight[:, :, 2] = weight[:, :, 2] - weight[:, :, 3]
                grad_weight[:, :, 1] = weight[:, :, 1] - weight[:, :, 2]
                grad_weight[:, :, 0] = -weight[:, :, 1]
            elif self.step == 2:
                grad_weight[:, :, 4] = weight[:, :, 4]
                grad_weight[:, :, 3] = weight[:, :, 3]
                grad_weight[:, :, 2] = weight[:, :, 2] - weight[:, :, 4]
                grad_weight[:, :, 1] = -weight[:, :, 3]
                grad_weight[:, :, 0] = -weight[:, :, 2]
        else:
            grad_weight = weight

        bias = self.conv.bias
        if bias is None:
            bias = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)

        return grad_weight, bias

    def forward(self, x):
        weight, bias = self.get_time_gradient_weight()
        return F.conv3d(
            x, weight, bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.groups
        )


class TDCR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 1, 1),
                 stride=(1, 1, 1), padding=(2, 0, 0), groups=1):
        super().__init__()

        self.l_tdc = nn.Sequential(
            TDC(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, step=-1),
            nn.BatchNorm3d(out_channels)
        )
        self.s_tdc = nn.Sequential(
            TDC(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, step=1),
            nn.BatchNorm3d(out_channels)
        )
        self.m_tdc = nn.Sequential(
            TDC(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, step=2),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        out = self.s_tdc(x) + self.m_tdc(x) + self.l_tdc(x)
        return F.relu(out, inplace=True)