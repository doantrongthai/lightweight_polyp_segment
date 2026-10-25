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


class ESDownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, 3, stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pool(x)
        x2 = self.conv(x)
        x1 = _resize_like(x1, x2)
        return self.relu(self.bn(torch.cat([x2, x1], 1)))


class ESUpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ESFCU(nn.Module):
    def __init__(self, chann, kernel_size, dropprob, dilated):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilated
        self.conv3x1_1 = nn.Conv2d(chann, chann, (kernel_size, 1), padding=((kernel_size - 1) // 2, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, kernel_size), padding=(0, (kernel_size - 1) // 2), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (kernel_size, 1), padding=(padding, 0), dilation=(dilated, 1), bias=True)
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, kernel_size), padding=(0, padding), dilation=(1, dilated), bias=True)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv3x1_1(x))
        out = self.relu(self.bn1(self.conv1x3_1(out)))
        out = self.relu(self.conv3x1_2(out))
        out = self.bn2(self.conv1x3_2(out))
        if self.dropout.p != 0:
            out = self.dropout(out)
        return F.relu(residual + out, inplace=True)


class ESPFCU(nn.Module):
    def __init__(self, chann):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv3x1_22 = nn.Conv2d(chann, chann, (3, 1), padding=(2, 0), dilation=(2, 1), bias=True)
        self.conv1x3_22 = nn.Conv2d(chann, chann, (1, 3), padding=(0, 2), dilation=(1, 2), bias=True)
        self.conv3x1_25 = nn.Conv2d(chann, chann, (3, 1), padding=(5, 0), dilation=(5, 1), bias=True)
        self.conv1x3_25 = nn.Conv2d(chann, chann, (1, 3), padding=(0, 5), dilation=(1, 5), bias=True)
        self.conv3x1_29 = nn.Conv2d(chann, chann, (3, 1), padding=(9, 0), dilation=(9, 1), bias=True)
        self.conv1x3_29 = nn.Conv2d(chann, chann, (1, 3), padding=(0, 9), dilation=(1, 9), bias=True)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-3)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv3x1_1(x))
        out = F.relu(self.bn1(self.conv1x3_1(out)))
        outs = []
        for c3, c1 in ((self.conv3x1_22, self.conv1x3_22), (self.conv3x1_25, self.conv1x3_25), (self.conv3x1_29, self.conv1x3_29)):
            branch = self.bn2(c1(F.relu(c3(out))))
            outs.append(self.dropout(branch) if self.dropout.p != 0 else branch)
        return F.relu(residual + outs[0] + outs[1] + outs[2], inplace=True)


@register_model("esnet")
class ESNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.initial_block = ESDownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(ESFCU(16, 3, 0.03, 1))
        self.layers.append(ESDownsamplerBlock(16, 64))
        for _ in range(2):
            self.layers.append(ESFCU(64, 5, 0.03, 1))
        self.layers.append(ESDownsamplerBlock(64, 128))
        for _ in range(3):
            self.layers.append(ESPFCU(128))
        self.layers.append(ESUpsamplerBlock(128, 64))
        self.layers.append(ESFCU(64, 5, 0, 1))
        self.layers.append(ESFCU(64, 5, 0, 1))
        self.layers.append(ESUpsamplerBlock(64, 16))
        self.layers.append(ESFCU(16, 3, 0, 1))
        self.layers.append(ESFCU(16, 3, 0, 1))
        self.output_conv = nn.ConvTranspose2d(16, classes, 2, stride=2, bias=True)

    def forward(self, x):
        size = x.shape[2:]
        out = self.initial_block(x)
        for layer in self.layers:
            out = layer(out)
        return _resize(self.output_conv(out), size)


__all__ = ["ESNet"]
