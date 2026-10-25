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


class LinkBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class LinkEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.block1 = LinkBasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = LinkBasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        return self.block2(self.block1(x))


class LinkDecoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super().__init__()
        mid = in_planes // 4
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, mid, 1, bias=bias), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(mid, mid, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(mid, out_planes, 1, bias=bias), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))

    def forward(self, x_high_level, x_low_level):
        x = self.tp_conv(self.conv1(x_high_level))
        x = _resize_like(x, x_low_level)
        return self.conv2(x)


@register_model("linknet")
class LinkNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.encoder1 = LinkEncoder(64, 64, 3, 1, 1)
        self.encoder2 = LinkEncoder(64, 128, 3, 2, 1)
        self.encoder3 = LinkEncoder(128, 256, 3, 2, 1)
        self.encoder4 = LinkEncoder(256, 512, 3, 2, 1)
        self.decoder4 = LinkDecoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = LinkDecoder(256, 128, 3, 2, 1, 1)
        self.decoder2 = LinkDecoder(128, 64, 3, 2, 1, 1)
        self.decoder1 = LinkDecoder(64, 64, 3, 1, 1, 0)
        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)

    def forward(self, x):
        size = x.shape[2:]
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)
        y = self.tp_conv2(self.conv2(self.tp_conv1(d1)))
        return _resize(y, size)


__all__ = ["LinkNet"]
