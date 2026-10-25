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


class SQFire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, 1)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, 1)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, 3, padding=1)
        self.relu2 = nn.ELU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        out = torch.cat([self.conv2(x), self.conv3(x)], 1)
        return self.relu2(out)


class SQParallelDilatedConv(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(inplanes, planes, 3, padding=d, dilation=d) for d in (1, 2, 3, 4)
        ])
        self.relus = nn.ModuleList([nn.ELU(inplace=True) for _ in range(4)])

    def forward(self, x):
        out = 0
        for conv, relu in zip(self.convs, self.relus):
            out = out + relu(conv(x))
        return out


@register_model("sqnet")
class SQNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.num_classes = classes
        self.conv1 = nn.Conv2d(3, 96, 3, stride=2, padding=1)
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.fire1_1 = SQFire(96, 16, 64)
        self.fire1_2 = SQFire(128, 16, 64)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fire2_1 = SQFire(128, 32, 128)
        self.fire2_2 = SQFire(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.fire3_1 = SQFire(256, 64, 256)
        self.fire3_2 = SQFire(512, 64, 256)
        self.fire3_3 = SQFire(512, 64, 256)
        self.parallel = SQParallelDilatedConv(512, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ELU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1)
        self.relu3 = nn.ELU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(256, 96, 3, stride=2, padding=1, output_padding=1)
        self.relu4 = nn.ELU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(192, classes, 3, stride=2, padding=1, output_padding=1)
        self.conv3_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv1_1 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv1_2 = nn.Conv2d(192, 192, 3, padding=1)
        self.relu1_1 = nn.ELU(inplace=True)
        self.relu1_2 = nn.ELU(inplace=True)
        self.relu2_1 = nn.ELU(inplace=True)
        self.relu2_2 = nn.ELU(inplace=True)
        self.relu3_1 = nn.ELU(inplace=True)
        self.relu3_2 = nn.ELU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        x_1 = self.relu1(self.conv1(x))
        x = self.maxpool1(x_1)
        x = self.fire1_1(x)
        x_2 = self.fire1_2(x)
        x = self.maxpool2(x_2)
        x = self.fire2_1(x)
        x_3 = self.fire2_2(x)
        x = self.maxpool3(x_3)
        x = self.fire3_1(x)
        x = self.fire3_2(x)
        x = self.fire3_3(x)
        x = self.parallel(x)
        y_3 = self.relu2(self.deconv1(x))
        x_3 = self.relu3_1(self.conv3_1(x_3))
        x_3 = _resize_like(x_3, y_3)
        x = self.relu3_2(self.conv3_2(torch.cat([x_3, y_3], 1)))
        y_2 = self.relu3(self.deconv2(x))
        x_2 = self.relu2_1(self.conv2_1(x_2))
        y_2 = _resize_like(y_2, x_2)
        x = self.relu2_2(self.conv2_2(torch.cat([x_2, y_2], 1)))
        y_1 = self.relu4(self.deconv3(x))
        x_1 = self.relu1_1(self.conv1_1(x_1))
        y_1 = _resize_like(y_1, x_1)
        x = self.relu1_2(self.conv1_2(torch.cat([x_1, y_1], 1)))
        return _resize(self.deconv4(x), size)


__all__ = ["SQNet"]
