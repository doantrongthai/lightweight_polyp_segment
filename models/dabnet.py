import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from . import register_model


def _resize(x, size, align_corners=True):
    if x.shape[2:] == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


def _resize_like(x, ref, align_corners=True):
    return _resize(x, ref.shape[2:], align_corners=align_corners)


class DABConv(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if bn_acti:
            self.bn_prelu = DABBNPReLU(n_out)

    def forward(self, x):
        out = self.conv(x)
        return self.bn_prelu(out) if self.bn_acti else out


class DABBNPReLU(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_in, eps=1e-3)
        self.acti = nn.PReLU(n_in)

    def forward(self, x):
        return self.acti(self.bn(x))


class DABModule(nn.Module):
    def __init__(self, n_in, d=1, k_size=3, dk_size=3):
        super().__init__()
        self.bn_relu_1 = DABBNPReLU(n_in)
        self.conv3x3 = DABConv(n_in, n_in // 2, k_size, 1, padding=1, bn_acti=True)
        self.dconv3x1 = DABConv(n_in // 2, n_in // 2, (dk_size, 1), 1, padding=(1, 0), groups=n_in // 2, bn_acti=True)
        self.dconv1x3 = DABConv(n_in // 2, n_in // 2, (1, dk_size), 1, padding=(0, 1), groups=n_in // 2, bn_acti=True)
        self.ddconv3x1 = DABConv(n_in // 2, n_in // 2, (dk_size, 1), 1, padding=(d, 0), dilation=(d, 1), groups=n_in // 2, bn_acti=True)
        self.ddconv1x3 = DABConv(n_in // 2, n_in // 2, (1, dk_size), 1, padding=(0, d), dilation=(1, d), groups=n_in // 2, bn_acti=True)
        self.bn_relu_2 = DABBNPReLU(n_in // 2)
        self.conv1x1 = DABConv(n_in // 2, n_in, 1, 1, padding=0)

    def forward(self, x):
        out = self.conv3x3(self.bn_relu_1(x))
        br1 = self.dconv1x3(self.dconv3x1(out))
        br2 = self.ddconv1x3(self.ddconv3x1(out))
        return self.conv1x1(self.bn_relu_2(br1 + br2)) + x


class DABDownSamplingBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        n_conv = n_out - n_in if n_in < n_out else n_out
        self.conv3x3 = DABConv(n_in, n_conv, 3, 2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = DABBNPReLU(n_out)

    def forward(self, x):
        out = self.conv3x3(x)
        if self.n_in < self.n_out:
            out = torch.cat([out, _resize_like(self.max_pool(x), out)], 1)
        return self.bn_prelu(out)


class DABInputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList([nn.AvgPool2d(3, stride=2, padding=1) for _ in range(ratio)])

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


@register_model("dabnet")
class DABNet(nn.Module):
    def __init__(self, classes=1, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            DABConv(3, 32, 3, 2, padding=1, bn_acti=True),
            DABConv(32, 32, 3, 1, padding=1, bn_acti=True),
            DABConv(32, 32, 3, 1, padding=1, bn_acti=True),
        )
        self.down_1 = DABInputInjection(1)
        self.down_2 = DABInputInjection(2)
        self.down_3 = DABInputInjection(3)
        self.bn_prelu_1 = DABBNPReLU(35)
        self.downsample_1 = DABDownSamplingBlock(35, 64)
        self.DAB_Block_1 = nn.Sequential(*[DABModule(64, d=2) for _ in range(block_1)])
        self.bn_prelu_2 = DABBNPReLU(131)
        self.downsample_2 = DABDownSamplingBlock(131, 128)
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.DAB_Block_2 = nn.Sequential(*[DABModule(128, d=dilation_block_2[i]) for i in range(block_2)])
        self.bn_prelu_3 = DABBNPReLU(259)
        self.classifier = DABConv(259, classes, 1, 1, padding=0)

    def forward(self, x):
        size = x.shape[2:]
        output0 = self.init_conv(x)
        down_1 = _resize_like(self.down_1(x), output0)
        down_2 = self.down_2(x)
        down_3 = self.down_3(x)
        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, _resize_like(down_2, output1)], 1))
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, _resize_like(down_3, output2)], 1))
        return _resize(self.classifier(output2_cat), size, align_corners=False)


__all__ = ["DABNet"]
