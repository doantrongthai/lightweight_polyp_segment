import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_model


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(x)
        x = torch.cat([conv_out, pool_out], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EDAModule(nn.Module):
    def __init__(self, in_channels, k, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, k, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(1, 0), bias=False)

        self.conv3 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(k)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(k, k, kernel_size=(3, 1), dilation=(dilation, 1), padding=(dilation, 0), bias=False)

        self.conv5 = nn.Conv2d(k, k, kernel_size=(1, 3), dilation=(1, dilation), padding=(0, dilation), bias=False)
        self.bn5 = nn.BatchNorm2d(k)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = torch.cat([x, residual], dim=1)
        return x


class EDABlock(nn.Module):
    def __init__(self, in_channels, k, num_block, dilations):
        super().__init__()
        assert len(dilations) == num_block

        self.modules_list = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_block):
            dilation = dilations[i]
            self.modules_list.append(EDAModule(current_channels, k, dilation))
            current_channels += k

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


@register_model("edanet")
class EDANetModel(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, k=40, num_b1=5, num_b2=8):
        super().__init__()

        self.num_classes = num_classes

        self.stage1 = DownsamplingBlock(in_channels, 15)

        self.stage2_down = DownsamplingBlock(15, 60)
        self.stage2 = EDABlock(60, k, num_b1, [1, 1, 1, 2, 2])

        self.stage3_down_conv = nn.Conv2d(260, 130, kernel_size=3, stride=2, padding=1, bias=False)
        self.stage3_down_bn = nn.BatchNorm2d(130)
        self.stage3_down_relu = nn.ReLU(inplace=True)
        self.stage3 = EDABlock(130, k, num_b2, [2, 2, 4, 4, 8, 8, 16, 16])

        self.project = nn.Conv2d(130 + k * num_b2, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]

        x = self.stage1(x)

        x = self.stage2_down(x)
        x = self.stage2(x)

        x = self.stage3_down_conv(x)
        x = self.stage3_down_bn(x)
        x = self.stage3_down_relu(x)
        x = self.stage3(x)

        x = self.project(x)
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)

        return x


def build_model(num_classes=1):
    return EDANetModel(num_classes=num_classes, in_channels=3)
