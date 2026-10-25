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


def fss_activation(name):
    if name == "PReLU":
        return nn.PReLU()
    if name == "ReLu6":
        return nn.ReLU6(inplace=True)
    return nn.ReLU(inplace=True)


class FSSInitialBlock(nn.Module):
    def __init__(self, ninput, noutput, non_linear="ReLU"):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput - ninput, eps=1e-3)
        self.relu = fss_activation(non_linear)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return torch.cat([out, _resize_like(self.pool(x), out)], 1)


class FSSDownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3, padding=0, dropout_prob=0.0, bias=False, non_linear="ReLU"):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.main_max1 = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(in_channels, out_channels, 1, bias=bias))
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, 2, stride=2, bias=bias), nn.BatchNorm2d(internal_channels), fss_activation(non_linear))
        self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, internal_channels, kernel_size, padding=padding, bias=bias), nn.BatchNorm2d(internal_channels), fss_activation(non_linear))
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, 1, bias=bias), nn.BatchNorm2d(out_channels), fss_activation(non_linear))
        self.ext_regul = nn.Dropout2d(dropout_prob)
        self.out_prelu = fss_activation(non_linear)

    def forward(self, x):
        main = self.main_max1(x)
        ext = self.ext_regul(self.ext_conv3(self.ext_conv2(self.ext_conv1(x))))
        main = _resize_like(main, ext)
        return self.out_prelu(main + ext)


class FSSUpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=2, padding=0, dropout_prob=0.0, bias=False, non_linear="ReLU"):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.main_conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=bias), nn.BatchNorm2d(out_channels))
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, 1, bias=bias), nn.BatchNorm2d(internal_channels), fss_activation(non_linear))
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size, stride=2, padding=padding, bias=bias),
            nn.BatchNorm2d(internal_channels),
            fss_activation(non_linear),
        )
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, 1, bias=bias), nn.BatchNorm2d(out_channels), fss_activation(non_linear))
        self.ext_regul = nn.Dropout2d(dropout_prob)
        self.out_prelu = fss_activation(non_linear)

    def forward(self, x, x_pre):
        x = _resize_like(x, x_pre)
        main = self.main_conv1(x + x_pre)
        main = F.interpolate(main, scale_factor=2, mode="bilinear", align_corners=True)
        ext = self.ext_regul(self.ext_conv3(self.ext_conv2(self.ext_conv1(x))))
        ext = _resize_like(ext, main)
        return self.out_prelu(main + ext)


class FSSDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout_prob=0.0, bias=False, non_linear="ReLU"):
        super().__init__()
        self.relu = fss_activation(non_linear)
        internal_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, internal_channels, 1, bias=bias)
        self.conv1_bn = nn.BatchNorm2d(internal_channels)
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size, stride, padding=int((kernel_size - 1) / 2 * dilation), dilation=dilation, bias=bias)
        self.conv2_bn = nn.BatchNorm2d(internal_channels)
        self.conv4 = nn.Conv2d(internal_channels, out_channels, 1, bias=bias)
        self.conv4_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1_bn(self.conv1(x)))
        out = self.relu(self.conv2_bn(self.conv2(out)))
        out = self.regul(self.conv4_bn(self.conv4(out)))
        return self.relu(out + residual)


class FSSFactorizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout_prob=0.0, bias=False, non_linear="ReLU"):
        super().__init__()
        self.relu = fss_activation(non_linear)
        internal_channels = in_channels // 4
        self.compress_conv1 = nn.Conv2d(in_channels, internal_channels, 1, bias=bias)
        self.conv1_bn = nn.BatchNorm2d(internal_channels)
        self.conv2_1 = nn.Conv2d(internal_channels, internal_channels, (kernel_size, 1), stride=(stride, 1), padding=(int((kernel_size - 1) / 2 * dilation), 0), dilation=(dilation, 1), bias=bias)
        self.conv2_1_bn = nn.BatchNorm2d(internal_channels)
        self.conv2_2 = nn.Conv2d(internal_channels, internal_channels, (1, kernel_size), stride=(1, stride), padding=(0, int((kernel_size - 1) / 2 * dilation)), dilation=(1, dilation), bias=bias)
        self.conv2_2_bn = nn.BatchNorm2d(internal_channels)
        self.extend_conv3 = nn.Conv2d(internal_channels, out_channels, 1, bias=bias)
        self.conv3_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1_bn(self.compress_conv1(x)))
        out = self.relu(self.conv2_1_bn(self.conv2_1(out)))
        out = self.relu(self.conv2_2_bn(self.conv2_2(out)))
        out = self.regul(self.conv3_bn(self.extend_conv3(out)))
        return self.relu(residual + out)


@register_model("fssnet")
class FSSNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.initial_block = FSSInitialBlock(3, 16)
        self.downsample1_0 = FSSDownsamplingBottleneck(16, 64, padding=1, dropout_prob=0.03)
        self.factorized1_1 = FSSFactorizedBlock(64, 64, dropout_prob=0.03)
        self.factorized1_2 = FSSFactorizedBlock(64, 64, dropout_prob=0.03)
        self.factorized1_3 = FSSFactorizedBlock(64, 64, dropout_prob=0.03)
        self.factorized1_4 = FSSFactorizedBlock(64, 64, dropout_prob=0.03)
        self.downsample2_0 = FSSDownsamplingBottleneck(64, 128, padding=1, dropout_prob=0.3)
        self.dilated2_1 = FSSDilatedBlock(128, 128, dilation=2, dropout_prob=0.3)
        self.dilated2_2 = FSSDilatedBlock(128, 128, dilation=5, dropout_prob=0.3)
        self.dilated2_3 = FSSDilatedBlock(128, 128, dilation=9, dropout_prob=0.3)
        self.dilated2_4 = FSSDilatedBlock(128, 128, dilation=2, dropout_prob=0.3)
        self.dilated2_5 = FSSDilatedBlock(128, 128, dilation=5, dropout_prob=0.3)
        self.dilated2_6 = FSSDilatedBlock(128, 128, dilation=9, dropout_prob=0.3)
        self.upsample4_0 = FSSUpsamplingBottleneck(128, 64, dropout_prob=0.3)
        self.bottleneck4_1 = FSSDilatedBlock(64, 64, dropout_prob=0.3)
        self.bottleneck4_2 = FSSDilatedBlock(64, 64, dropout_prob=0.3)
        self.upsample5_0 = FSSUpsamplingBottleneck(64, 16, dropout_prob=0.3)
        self.bottleneck5_1 = FSSDilatedBlock(16, 16, dropout_prob=0.3)
        self.bottleneck5_2 = FSSDilatedBlock(16, 16, dropout_prob=0.3)
        self.transposed_conv = nn.ConvTranspose2d(16, classes, 3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        size = x.shape[2:]
        x = self.initial_block(x)
        x_1 = self.downsample1_0(x)
        x = self.factorized1_4(self.factorized1_3(self.factorized1_2(self.factorized1_1(x_1))))
        x_2 = self.downsample2_0(x)
        x = self.dilated2_6(self.dilated2_5(self.dilated2_4(self.dilated2_3(self.dilated2_2(self.dilated2_1(x_2))))))
        x = self.bottleneck4_2(self.bottleneck4_1(self.upsample4_0(x, x_2)))
        x = self.bottleneck5_2(self.bottleneck5_1(self.upsample5_0(x, x_1)))
        return _resize(self.transposed_conv(x), size)


__all__ = ["FSSNet"]
