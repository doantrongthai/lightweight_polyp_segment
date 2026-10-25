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


class ERFDownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, 3, stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        conv = self.conv(x)
        pool = _resize_like(self.pool(x), conv)
        return F.relu(self.bn(torch.cat([conv, pool], 1)))


class ERFNonBottleneck1D(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-3)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), padding=(dilated, 0), dilation=(dilated, 1), bias=True)
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), padding=(0, dilated), dilation=(1, dilated), bias=True)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-3)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        out = F.relu(self.conv3x1_1(x))
        out = F.relu(self.bn1(self.conv1x3_1(out)))
        out = F.relu(self.conv3x1_2(out))
        out = self.bn2(self.conv1x3_2(out))
        if self.dropout.p != 0:
            out = self.dropout(out)
        return F.relu(out + x)


class ERFEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = ERFDownsamplerBlock(3, 16)
        self.layers = nn.ModuleList([ERFDownsamplerBlock(16, 64)])
        for _ in range(5):
            self.layers.append(ERFNonBottleneck1D(64, 0.03, 1))
        self.layers.append(ERFDownsamplerBlock(64, 128))
        for _ in range(2):
            for d in (2, 4, 8, 16):
                self.layers.append(ERFNonBottleneck1D(128, 0.3, d))
        self.output_conv = nn.Conv2d(128, num_classes, 1)

    def forward(self, x, predict=False):
        out = self.initial_block(x)
        for layer in self.layers:
            out = layer(out)
        return self.output_conv(out) if predict else out


class ERFUpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ERFDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList([
            ERFUpsamplerBlock(128, 64),
            ERFNonBottleneck1D(64, 0, 1),
            ERFNonBottleneck1D(64, 0, 1),
            ERFUpsamplerBlock(64, 16),
            ERFNonBottleneck1D(16, 0, 1),
            ERFNonBottleneck1D(16, 0, 1),
        ])
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, bias=True)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return self.output_conv(out)


@register_model("erfnet")
class ERFNet(nn.Module):
    def __init__(self, classes=1, encoder=None):
        super().__init__()
        self.encoder = ERFEncoder(classes) if encoder is None else encoder
        self.decoder = ERFDecoder(classes)

    def forward(self, x, only_encode=False):
        size = x.shape[2:]
        if only_encode:
            return self.encoder.forward(x, predict=True)
        return _resize(self.decoder(self.encoder(x)), size)


__all__ = ["ERFNet"]
