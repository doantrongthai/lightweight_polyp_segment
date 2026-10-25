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


class FastPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.conv1 = ConvBNReLU(in_channels, inter_channels, 1)
        self.conv2 = ConvBNReLU(in_channels, inter_channels, 1)
        self.conv3 = ConvBNReLU(in_channels, inter_channels, 1)
        self.conv4 = ConvBNReLU(in_channels, inter_channels, 1)
        self.out = ConvBNReLU(in_channels * 2, out_channels, 1)

    def forward(self, x):
        size = x.shape[2:]
        feat1 = _resize(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = _resize(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = _resize(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = _resize(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        return self.out(torch.cat([x, feat1, feat2, feat3, feat4], dim=1))


class FastLearningToDownsample(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super().__init__()
        self.conv = ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        return self.dsconv2(self.dsconv1(self.conv(x)))


class FastGlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128, t=6, num_blocks=(3, 3, 3)):
        super().__init__()
        self.bottleneck1 = self._make_layer(in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = FastPyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, inplanes, planes, blocks, t=6, stride=1):
        layers = [LinearBottleneck(inplanes, planes, t, stride)]
        for _ in range(1, blocks):
            layers.append(LinearBottleneck(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.ppm(self.bottleneck3(self.bottleneck2(self.bottleneck1(x))))


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


@register_model("fastscnn")
class FastSCNN(nn.Module):
    def __init__(self, classes=1, aux=False):
        super().__init__()
        self.aux = aux
        self.learning_to_downsample = FastLearningToDownsample(32, 48, 64)
        self.global_feature_extractor = FastGlobalFeatureExtractor(64, (64, 96, 128), 128, 6, (3, 3, 3))
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = FastClassifier(128, classes)
        if aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, classes, 1),
            )

    def forward(self, x):
        size = x.shape[2:]
        higher_res_features = self.learning_to_downsample(x)
        out = self.global_feature_extractor(higher_res_features)
        out = self.feature_fusion(higher_res_features, out)
        out = _resize(self.classifier(out), size)
        if self.aux:
            return out, _resize(self.auxlayer(higher_res_features), size)
        return out


__all__ = ["FastSCNN"]
