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


def fpe_conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def fpe_conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=bias)


class FPESEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        return x * y


class FPEBlock(nn.Module):
    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=1, t=1, scales=4, se=False, norm_layer=None):
        super().__init__()
        if inplanes % scales != 0:
            raise ValueError("Planes must be divisible by scales")
        norm_layer = norm_layer or nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = fpe_conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([
            fpe_conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=bottleneck_planes // scales, dilation=dilat[i], padding=dilat[i])
            for i in range(scales)
        ])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)])
        self.conv3 = fpe_conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = FPESEModule(outplanes) if se else None
        self.downsample = downsample
        self.scales = scales

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            inp = xs[s] if s == 0 else xs[s] + ys[-1]
            ys.append(self.relu(self.bn2[s](self.conv2[s](inp))))
        out = self.bn3(self.conv3(torch.cat(ys, 1)))
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        identity = _resize_like(identity, out)
        return self.relu(out + identity)


class FPEMEUModule(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out):
        super().__init__()
        self.conv1x1_low = nn.Conv2d(channels_low, channel_out, 1, bias=False)
        self.bn_low = nn.BatchNorm2d(channel_out)
        self.sa_conv = nn.Conv2d(1, 1, 1, bias=False)
        self.conv1x1_high = nn.Conv2d(channels_high, channel_out, 1, bias=False)
        self.bn_high = nn.BatchNorm2d(channel_out)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(channel_out, channel_out, 1, bias=False)
        self.sa_sigmoid = nn.Sigmoid()
        self.ca_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        fms_low = self.bn_low(self.conv1x1_low(fms_low))
        sa = self.sa_sigmoid(self.sa_conv(torch.mean(fms_low, dim=1, keepdim=True)))
        fms_high = self.bn_high(self.conv1x1_high(fms_high))
        ca = self.ca_sigmoid(self.relu(self.ca_conv(self.avg_pool(fms_high))))
        fms_high = _resize_like(fms_high, fms_low)
        return ca * fms_low + sa * fms_high


@register_model("fpenet")
class FPENet(nn.Module):
    def __init__(self, classes=1, zero_init_residual=False, width=16, scales=4, se=False, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        outplanes = [int(width * 2 ** i) for i in range(3)]
        self.block_num = [1, 3, 9]
        self.dilation = [1, 2, 4, 8]
        self.inplanes = outplanes[0]
        self.conv1 = nn.Conv2d(3, outplanes[0], 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(outplanes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(outplanes[0], self.block_num[0], self.dilation, 1, 1, scales, se, norm_layer)
        self.layer2 = self._make_layer(outplanes[1], self.block_num[1], self.dilation, 2, 4, scales, se, norm_layer)
        self.layer3 = self._make_layer(outplanes[2], self.block_num[2], self.dilation, 2, 4, scales, se, norm_layer)
        self.meu1 = FPEMEUModule(64, 32, 64)
        self.meu2 = FPEMEUModule(64, 16, 32)
        self.project_layer = nn.Conv2d(32, classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FPEBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, planes, blocks, dilation, stride=1, t=1, scales=4, se=False, norm_layer=None):
        norm_layer = norm_layer or nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(fpe_conv1x1(self.inplanes, planes, stride), norm_layer(planes))
        layers = [FPEBlock(self.inplanes, planes, dilation, downsample, stride, t, scales, se, norm_layer)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(FPEBlock(self.inplanes, planes, dilation, scales=scales, se=se, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.shape[2:]
        x = self.relu(self.bn1(self.conv1(x)))
        x_1 = self.layer1(x)
        x_2_0 = self.layer2[0](x_1)
        x_2_1 = self.layer2[1](x_2_0)
        x_2_2 = self.layer2[2](x_2_1)
        x_2 = x_2_0 + x_2_2
        x_3_0 = self.layer3[0](x_2)
        x_3 = x_3_0
        for i in range(1, 9):
            x_3 = self.layer3[i](x_3)
        x_3 = x_3_0 + x_3
        x2 = self.meu1(x_3, x_2)
        x1 = self.meu2(x2, x_1)
        return _resize(self.project_layer(x1), size)


__all__ = ["FPENet"]
