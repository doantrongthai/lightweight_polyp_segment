import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


def _resize_like(x, ref, mode="bilinear", align_corners=True):
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode=mode, align_corners=align_corners)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    """Upsample + 3x3 conv (giống UpConv gốc, KHÔNG dùng ConvTranspose2d)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (Hu et al.): global avg pool -> FC giam -> ReLU
    -> FC mo rong -> Sigmoid -> nhan lai vao feature map theo tung kenh."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.avg_pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s, s


class SEUpBlock(nn.Module):
    """Up-conv + SE-recalibrate skip connection + concat + conv."""

    def __init__(self, in_channels, skip_channels, out_channels, reduction=16):
        super().__init__()
        self.up = UpConv(in_channels, skip_channels)
        self.se = SEBlock(skip_channels, reduction=reduction)
        self.conv = ConvBlock(skip_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = _resize_like(x, skip)
        skip_se, gate = self.se(skip)
        x = self.conv(torch.cat([skip_se, x], 1))
        return x, gate


@register_model("unet_se")
class UNetSE(nn.Module):
    """U-Net với SE attention gan vao skip connection (thay vi attention gate)."""

    def __init__(self, classes=1, in_channels=3, feature_scale=1, reduction=16):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [max(f // feature_scale, 8) for f in filters]

        self.conv1 = ConvBlock(in_channels, filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.conv4 = ConvBlock(filters[2], filters[3])
        self.center = ConvBlock(filters[3], filters[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = SEUpBlock(filters[4], filters[3], filters[3], reduction=reduction)
        self.up3 = SEUpBlock(filters[3], filters[2], filters[2], reduction=reduction)
        self.up2 = SEUpBlock(filters[2], filters[1], filters[1], reduction=reduction)
        self.up1 = SEUpBlock(filters[1], filters[0], filters[0], reduction=reduction)

        self.classifier = nn.Conv2d(filters[0], classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool(c1))
        c3 = self.conv3(self.pool(c2))
        c4 = self.conv4(self.pool(c3))
        center = self.center(self.pool(c4))

        u4, _ = self.up4(center, c4)
        u3, _ = self.up3(u4, c3)
        u2, _ = self.up2(u3, c2)
        u1, _ = self.up1(u2, c1)

        out = self.classifier(u1)
        return _resize_like(out, x) if out.shape[2:] != size else out


__all__ = ["UNetSE"]
