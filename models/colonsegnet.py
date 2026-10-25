import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_model


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            SELayer(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class StridedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual_block1 = ResidualBlock(in_channels, out_channels)
        self.strided_conv = StridedConvBlock(out_channels, out_channels)
        self.residual_block2 = ResidualBlock(out_channels, out_channels)
        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip_high = self.residual_block1(x)
        skip_low = self.residual_block2(self.strided_conv(skip_high))
        pooled = self.pooling(skip_low)
        return skip_high, skip_low, pooled


@register_model("colonsegnet")
@register_model("compnet")
class ColonSegNet(nn.Module):
    """
    Faithful re-implementation of the official ColonSegNet / CompNet
    (Jha et al., "Real-Time Polyp Detection, Localization and Segmentation
    in Colonoscopy Using Deep Learning", IEEE Access 2021).
    Reference: https://github.com/DebeshJha/ColonSegNet/blob/main/model.py

    Default input resolution used here is 352x352 (instead of the paper's
    512x512). All strided-conv / transpose-conv arithmetic in this
    architecture divides 352 cleanly at every stage (352 -> 176 -> 88 -> 44),
    so the decoder's transpose convolutions land on the exact same spatial
    size as their corresponding skip connections -- no interpolation is
    ever actually triggered for a 352x352 input; the resize fallback below
    is kept purely as a safety net for other resolutions.
    """

    def __init__(self, classes=1):
        super().__init__()
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 256)

        # Decoder 1
        self.t1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, padding=0)
        self.r1 = ResidualBlock(192, 128)
        self.t2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.r2 = ResidualBlock(256, 128)

        # Decoder 2
        self.t3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.r3 = ResidualBlock(128, 64)
        self.t4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.r4 = ResidualBlock(96, 32)

        self.output = nn.Conv2d(32, classes, 1)

    @staticmethod
    def _cat_resize(x, skip):
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        size = x.shape[2:]

        s11, s12, p1 = self.e1(x)      # 352, 176, 88
        s21, s22, _ = self.e2(p1)      # 88, 44, (22 unused, matches official)

        # cat([t1, s12])  -- transpose output first, matches official order
        t1 = self._cat_resize(self.t1(s22), s12)
        r1 = self.r1(t1)

        # cat([r1, t2])   -- residual output first, matches official order
        t2 = self._cat_resize(r1, self.t2(s21))
        r2 = self.r2(t2)

        # cat([t3, s11])  -- transpose output first, matches official order
        t3 = self._cat_resize(self.t3(r2), s11)
        r3 = self.r3(t3)

        # cat([r3, t4])   -- residual output first, matches official order
        t4 = self._cat_resize(r3, self.t4(s12))
        r4 = self.r4(t4)

        out = self.output(r4)
        if out.shape[2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=True)
        return out
