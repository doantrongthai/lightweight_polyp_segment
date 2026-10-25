import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


def _resize_like(x, ref, mode="bilinear", align_corners=True):
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode=mode, align_corners=align_corners)


class AttnConvBlock(nn.Module):
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


class AttnUpsample(nn.Module):
    """Upsample + 3x3 conv (matches original UpConv, NOT ConvTranspose2d)."""

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


class AttnGate(nn.Module):
    """Additive attention gate (Oktay et al.), no sub-sampling, gate == skip resolution."""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.w_gate(gate)
        s = self.w_skip(skip)
        g = _resize_like(g, s)
        attn = self.psi(self.relu(g + s))
        attn = _resize_like(attn, skip)
        return skip * attn, attn


class AttnUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(skip_channels // 2, 1)
        self.up = AttnUpsample(in_channels, skip_channels)
        self.gate = AttnGate(skip_channels, skip_channels, inter_channels)
        self.conv = AttnConvBlock(skip_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = _resize_like(x, skip)
        gated_skip, attn = self.gate(x, skip)
        x = self.conv(torch.cat([gated_skip, x], 1))
        return x, attn


@register_model("attention_unet")
class AttentionUNet(nn.Module):
    def __init__(self, classes=1, in_channels=3, feature_scale=1):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [max(f // feature_scale, 8) for f in filters]

        self.conv1 = AttnConvBlock(in_channels, filters[0])
        self.conv2 = AttnConvBlock(filters[0], filters[1])
        self.conv3 = AttnConvBlock(filters[1], filters[2])
        self.conv4 = AttnConvBlock(filters[2], filters[3])
        self.center = AttnConvBlock(filters[3], filters[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = AttnUpBlock(filters[4], filters[3], filters[3], inter_channels=max(filters[3] // 2, 1))
        self.up3 = AttnUpBlock(filters[3], filters[2], filters[2], inter_channels=max(filters[2] // 2, 1))
        self.up2 = AttnUpBlock(filters[2], filters[1], filters[1], inter_channels=max(filters[1] // 2, 1))
        self.up1 = AttnUpBlock(filters[1], filters[0], filters[0], inter_channels=max(filters[0] // 2, 1))

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


__all__ = ["AttentionUNet"]
