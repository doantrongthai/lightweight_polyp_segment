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


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class DSConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, in_channels * t, 1),
            DWConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_shortcut else out


class FeatureFusionModule(nn.Module):
    def __init__(self, higher_in_channels, lower_in_channels, out_channels):
        super().__init__()
        self.dwconv = DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.conv_higher_res = nn.Sequential(nn.Conv2d(higher_in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = _resize_like(lower_res_feature, higher_res_feature)
        lower_res_feature = self.conv_lower_res(self.dwconv(lower_res_feature))
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        return self.relu(higher_res_feature + lower_res_feature)


class FastClassifier(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1):
        super().__init__()
        self.dsconv1 = DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(dw_channels, num_classes, 1))

    def forward(self, x):
        return self.conv(self.dsconv2(self.dsconv1(x)))


class ContextShallowNet(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=64, out_channels=128):
        super().__init__()
        self.conv = ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DSConv(dw_channels2, out_channels, 2)
        self.dsconv3 = DSConv(out_channels, out_channels, 1)

    def forward(self, x):
        return self.dsconv3(self.dsconv2(self.dsconv1(self.conv(x))))


class ContextDeepNet(nn.Module):
    def __init__(self, in_channels, block_channels, t, num_blocks):
        super().__init__()
        self.conv_ = ConvBNReLU(3, in_channels, 3, 2)
        self.bottleneck1 = self._layer(in_channels, block_channels[0], num_blocks[0], t[0], 1)
        self.bottleneck2 = self._layer(block_channels[0], block_channels[1], num_blocks[1], t[1], 1)
        self.bottleneck3 = self._layer(block_channels[1], block_channels[2], num_blocks[2], t[2], 2)
        self.bottleneck4 = self._layer(block_channels[2], block_channels[3], num_blocks[3], t[3], 2)
        self.bottleneck5 = self._layer(block_channels[3], block_channels[4], num_blocks[4], t[4], 1)
        self.bottleneck6 = self._layer(block_channels[4], block_channels[5], num_blocks[5], t[5], 1)

    def _layer(self, in_channels, out_channels, blocks, t, stride):
        layers = [LinearBottleneck(in_channels, out_channels, t, stride)]
        for _ in range(1, blocks):
            layers.append(LinearBottleneck(out_channels, out_channels, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        return self.bottleneck6(x)


@register_model("contextnet")
class ContextNet(nn.Module):
    def __init__(self, classes=1, aux=False):
        super().__init__()
        self.aux = aux
        self.spatial_detail = ContextShallowNet(32, 64, 128)
        self.context_feature_extractor = ContextDeepNet(32, [32, 32, 48, 64, 96, 128], [1, 6, 6, 6, 6, 6], [1, 1, 3, 3, 2, 2])
        self.feature_fusion = FeatureFusionModule(128, 128, 128)
        self.classifier = FastClassifier(128, classes)
        if aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(128, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, classes, 1),
            )

    def forward(self, x):
        size = x.shape[2:]
        higher = self.spatial_detail(x)
        x_low = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=True)
        low = self.context_feature_extractor(x_low)
        out = self.feature_fusion(higher, low)
        out = _resize(self.classifier(out), size)
        if self.aux:
            return out, _resize(self.auxlayer(higher), size)
        return out


__all__ = ["ContextNet"]
