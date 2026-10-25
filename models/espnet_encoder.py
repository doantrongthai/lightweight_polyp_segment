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


class ESPCBR(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(n_out, eps=1e-3)
        self.act = nn.PReLU(n_out)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ESPBR(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_out, eps=1e-3)
        self.act = nn.PReLU(n_out)

    def forward(self, x):
        return self.act(self.bn(x))


class ESPC(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x)


class ESPCDilated(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1, d=1):
        super().__init__()
        padding = ((k_size - 1) // 2) * d
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, dilation=d, bias=False)

    def forward(self, x):
        return self.conv(x)


class ESPDownSamplerB(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        n = n_out // 5
        n1 = n_out - 4 * n
        self.c1 = ESPC(n_in, n, 3, 2)
        self.d1 = ESPCDilated(n, n1, 3, 1, 1)
        self.d2 = ESPCDilated(n, n, 3, 1, 2)
        self.d4 = ESPCDilated(n, n, 3, 1, 4)
        self.d8 = ESPCDilated(n, n, 3, 1, 8)
        self.d16 = ESPCDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(n_out, eps=1e-3)
        self.act = nn.PReLU(n_out)

    def forward(self, x):
        out = self.c1(x)
        d1 = self.d1(out)
        d2 = self.d2(out)
        d4 = d2 + self.d4(out)
        d8 = d4 + self.d8(out)
        d16 = d8 + self.d16(out)
        return self.act(self.bn(torch.cat([d1, d2, d4, d8, d16], 1)))


class ESPBlock(nn.Module):
    def __init__(self, n_in, n_out, add=True):
        super().__init__()
        n = n_out // 5
        n1 = n_out - 4 * n
        self.c1 = ESPC(n_in, n, 1)
        self.d1 = ESPCDilated(n, n1, 3, 1, 1)
        self.d2 = ESPCDilated(n, n, 3, 1, 2)
        self.d4 = ESPCDilated(n, n, 3, 1, 4)
        self.d8 = ESPCDilated(n, n, 3, 1, 8)
        self.d16 = ESPCDilated(n, n, 3, 1, 16)
        self.bn = ESPBR(n_out)
        self.add = add

    def forward(self, x):
        out = self.c1(x)
        d1 = self.d1(out)
        d2 = self.d2(out)
        d4 = d2 + self.d4(out)
        d8 = d4 + self.d8(out)
        d16 = d8 + self.d16(out)
        combine = torch.cat([d1, d2, d4, d8, d16], 1)
        if self.add:
            combine = x + combine
        return self.bn(combine)


class ESPInputProjectionA(nn.Module):
    def __init__(self, sampling_times):
        super().__init__()
        self.pool = nn.ModuleList([nn.AvgPool2d(3, stride=2, padding=1) for _ in range(sampling_times)])

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


@register_model("espnet_encoder")
class ESPNet_Encoder(nn.Module):
    def __init__(self, classes=1, p=5, q=3):
        super().__init__()
        self.level1 = ESPCBR(3, 16, 3, 2)
        self.sample1 = ESPInputProjectionA(1)
        self.sample2 = ESPInputProjectionA(2)
        self.b1 = ESPBR(19)
        self.level2_0 = ESPDownSamplerB(19, 64)
        self.level2 = nn.ModuleList([ESPBlock(64, 64) for _ in range(p)])
        self.b2 = ESPBR(131)
        self.level3_0 = ESPDownSamplerB(131, 128)
        self.level3 = nn.ModuleList([ESPBlock(128, 128) for _ in range(q)])
        self.b3 = ESPBR(256)
        self.classifier = ESPC(256, classes, 1)

    def forward(self, x):
        size = x.shape[2:]
        output0 = self.level1(x)
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)
        output0_cat = self.b1(torch.cat([output0, _resize_like(inp1, output0)], 1))
        output1_0 = self.level2_0(output0_cat)
        output1 = output1_0
        for layer in self.level2:
            output1 = layer(output1)
        output1_cat = self.b2(torch.cat([output1, output1_0, _resize_like(inp2, output1)], 1))
        output2_0 = self.level3_0(output1_cat)
        output2 = output2_0
        for layer in self.level3:
            output2 = layer(output2)
        output2_cat = self.b3(torch.cat([output2_0, output2], 1))
        return _resize(self.classifier(output2_cat), size)


__all__ = ["ESPNet_Encoder"]
