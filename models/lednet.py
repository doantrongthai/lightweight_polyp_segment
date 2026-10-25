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


def led_split(x):
    c1 = round(int(x.size(1)) * 0.5)
    return x[:, :c1].contiguous(), x[:, c1:].contiguous()


def led_channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)


class LEDConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LEDDownsamplerBlock(nn.Module):
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


class LEDSSNbtModule(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        oup_inc = chann // 2
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), padding=(1, 0), bias=True)
        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), padding=(0, 1), bias=True)
        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-3)
        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), padding=(dilated, 0), dilation=(dilated, 1), bias=True)
        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), padding=(0, dilated), dilation=(1, dilated), bias=True)
        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-3)
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), padding=(1, 0), bias=True)
        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), padding=(0, 1), bias=True)
        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-3)
        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), padding=(dilated, 0), dilation=(dilated, 1), bias=True)
        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), padding=(0, dilated), dilation=(1, dilated), bias=True)
        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        residual = x
        x1, x2 = led_split(x)
        output1 = self.relu(self.bn1_l(self.conv1x3_1_l(self.relu(self.conv3x1_1_l(x1)))))
        output2 = self.relu(self.bn1_r(self.conv3x1_1_r(self.relu(self.conv1x3_1_r(x2)))))
        output1 = self.bn2_l(self.conv1x3_2_l(self.relu(self.conv3x1_2_l(output1))))
        output2 = self.bn2_r(self.conv3x1_2_r(self.relu(self.conv1x3_2_r(output2))))
        if self.dropout.p != 0:
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)
        out = F.relu(residual + torch.cat([output1, output2], 1))
        return led_channel_shuffle(out, 2)


class LEDAPNModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), LEDConvBnRelu(in_ch, out_ch, 1))
        self.mid = LEDConvBnRelu(in_ch, out_ch, 1)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, 1, (7, 1), stride=(2, 1), padding=(3, 0), bias=True),
            nn.Conv2d(1, 1, (1, 7), stride=(1, 2), padding=(0, 3), bias=True),
            nn.BatchNorm2d(1, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(1, 1, (5, 1), stride=(2, 1), padding=(2, 0), bias=True),
            nn.Conv2d(1, 1, (1, 5), stride=(1, 2), padding=(0, 2), bias=True),
            nn.BatchNorm2d(1, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1, 1, (3, 1), stride=(2, 1), padding=(1, 0), bias=True),
            nn.Conv2d(1, 1, (1, 3), stride=(1, 2), padding=(0, 1), bias=True),
            nn.BatchNorm2d(1, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, (3, 1), padding=(1, 0), bias=True),
            nn.Conv2d(1, 1, (1, 3), padding=(0, 1), bias=True),
            nn.BatchNorm2d(1, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, (5, 1), padding=(2, 0), bias=True),
            nn.Conv2d(1, 1, (1, 5), padding=(0, 2), bias=True),
            nn.BatchNorm2d(1, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, (7, 1), padding=(3, 0), bias=True),
            nn.Conv2d(1, 1, (1, 7), padding=(0, 3), bias=True),
            nn.BatchNorm2d(1, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        b1 = _resize(self.branch1(x), (h, w))
        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = _resize(self.down3(x2), ((h + 3) // 4, (w + 3) // 4))
        x = _resize(self.conv2(x2) + x3, ((h + 1) // 2, (w + 1) // 2))
        x = _resize(x + self.conv1(x1), (h, w))
        return torch.mul(x, mid) + b1


@register_model("lednet")
class LEDNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.initial_block = LEDDownsamplerBlock(3, 32)
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(LEDSSNbtModule(32, 0.03, 1))
        self.layers.append(LEDDownsamplerBlock(32, 64))
        for _ in range(2):
            self.layers.append(LEDSSNbtModule(64, 0.03, 1))
        self.layers.append(LEDDownsamplerBlock(64, 128))
        for d in (1, 2, 5, 9, 2, 5, 9, 17):
            self.layers.append(LEDSSNbtModule(128, 0.3, d))
        self.apn = LEDAPNModule(128, classes)

    def forward(self, x):
        size = x.shape[2:]
        out = self.initial_block(x)
        for layer in self.layers:
            out = layer(out)
        return _resize(self.apn(out), size)


__all__ = ["LEDNet"]
