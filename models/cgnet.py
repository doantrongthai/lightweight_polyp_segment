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


class CGConvBNPReLU(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(n_out, eps=1e-3)
        self.act = nn.PReLU(n_out)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CGBNPReLU(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_out, eps=1e-3)
        self.act = nn.PReLU(n_out)

    def forward(self, x):
        return self.act(self.bn(x))


class CGConv(nn.Module):
    def __init__(self, n_in, n_out, k_size, stride=1, groups=1, dilation=1):
        super().__init__()
        padding = ((k_size - 1) // 2) * dilation
        self.conv = nn.Conv2d(n_in, n_out, k_size, stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


class CGFGlo(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class CGContextGuidedBlockDown(nn.Module):
    def __init__(self, n_in, n_out, dilation_rate=2, reduction=16):
        super().__init__()
        self.conv1x1 = CGConvBNPReLU(n_in, n_out, 3, 2)
        self.F_loc = CGConv(n_out, n_out, 3, groups=n_out)
        self.F_sur = CGConv(n_out, n_out, 3, groups=n_out, dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(2 * n_out, eps=1e-3)
        self.act = nn.PReLU(2 * n_out)
        self.reduce = CGConv(2 * n_out, n_out, 1)
        self.F_glo = CGFGlo(n_out, reduction)

    def forward(self, x):
        out = self.conv1x1(x)
        loc = self.F_loc(out)
        sur = self.F_sur(out)
        joint = self.reduce(self.act(self.bn(torch.cat([loc, sur], 1))))
        return self.F_glo(joint)


class CGContextGuidedBlock(nn.Module):
    def __init__(self, n_in, n_out, dilation_rate=2, reduction=16, add=True):
        super().__init__()
        n = n_out // 2
        self.conv1x1 = CGConvBNPReLU(n_in, n, 1)
        self.F_loc = CGConv(n, n, 3, groups=n)
        self.F_sur = CGConv(n, n, 3, groups=n, dilation=dilation_rate)
        self.bn_prelu = CGBNPReLU(n_out)
        self.add = add
        self.F_glo = CGFGlo(n_out, reduction)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.bn_prelu(torch.cat([self.F_loc(out), self.F_sur(out)], 1))
        out = self.F_glo(out)
        return x + out if self.add else out


class CGInputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList([nn.AvgPool2d(3, stride=2, padding=1) for _ in range(ratio)])

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


@register_model("cgnet")
class CGNet(nn.Module):
    def __init__(self, classes=1, M=3, N=21, dropout_flag=False):
        super().__init__()
        self.level1_0 = CGConvBNPReLU(3, 32, 3, 2)
        self.level1_1 = CGConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = CGConvBNPReLU(32, 32, 3, 1)
        self.sample1 = CGInputInjection(1)
        self.sample2 = CGInputInjection(2)
        self.b1 = CGBNPReLU(35)
        self.level2_0 = CGContextGuidedBlockDown(35, 64, dilation_rate=2, reduction=8)
        self.level2 = nn.ModuleList([CGContextGuidedBlock(64, 64, dilation_rate=2, reduction=8) for _ in range(M - 1)])
        self.bn_prelu_2 = CGBNPReLU(131)
        self.level3_0 = CGContextGuidedBlockDown(131, 128, dilation_rate=4, reduction=16)
        self.level3 = nn.ModuleList([CGContextGuidedBlock(128, 128, dilation_rate=4, reduction=16) for _ in range(N - 1)])
        self.bn_prelu_3 = CGBNPReLU(256)
        if dropout_flag:
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), CGConv(256, classes, 1))
        else:
            self.classifier = CGConv(256, classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        output0 = self.level1_2(self.level1_1(self.level1_0(x)))
        inp1 = _resize_like(self.sample1(x), output0)
        inp2 = self.sample2(x)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        output1 = output1_0
        for layer in self.level2:
            output1 = layer(output1)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, _resize_like(inp2, output1)], 1))
        output2_0 = self.level3_0(output1_cat)
        output2 = output2_0
        for layer in self.level3:
            output2 = layer(output2)
        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
        return _resize(self.classifier(output2_cat), size, align_corners=False)


__all__ = ["CGNet"]
