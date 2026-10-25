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


class UNetDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), UNetDoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.mpconv(x)


class UNetUp(nn.Module):
    """
    FIX vs previous version:
      1. Bug: self.up = ConvTranspose2d(in_ch//2, in_ch//2, ...) kept the
         channel count unchanged (only upsampled spatially). The standard
         UNet transpose-conv both upsamples AND halves the channel count
         (in_ch -> in_ch//2) so that concatenation with the same-size skip
         connection yields exactly `in_ch` channels again.
      2. Bug: forward() always took the bilinear-interpolation branch
         (bilinear=True by default, never overridden), so the
         ConvTranspose2d layer above was constructed but never actually
         called in forward() -- dead weights that inflated the parameter
         count without doing anything. The reference architecture always
         uses a learned ConvTranspose2d, never bilinear interpolation, so
         that branch is removed entirely.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = UNetDoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = _resize_like(x1, x2)  # safety net if H/W ever mismatch by 1px
        return self.conv(torch.cat([x2, x1], dim=1))


@register_model("unet")
class UNet(nn.Module):
    def __init__(self, classes=1, input_size=None):
        super().__init__()
        # Same math guard as the reference UNetModel: 4 MaxPool(stride=2)
        # stages require the spatial size to be divisible by 2**4 = 16.
        if input_size is not None and input_size % 16 != 0:
            raise ValueError(
                f"UNet requires input_size divisible by 16. Got {input_size}."
            )

        self.inc = UNetDoubleConv(3, 64)
        self.down1 = UNetDown(64, 128)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        self.down4 = UNetDown(512, 1024)   # FIX: bottleneck now doubles channels (512->1024)

        self.up1 = UNetUp(1024, 512)       # FIX: was UNetUp(1024, 256)
        self.up2 = UNetUp(512, 256)        # FIX: was UNetUp(512, 128)
        self.up3 = UNetUp(256, 128)        # FIX: was UNetUp(256, 64)
        self.up4 = UNetUp(128, 64)         # FIX: was UNetUp(128, 64) -- in_ch was already
                                            # right by coincidence, kept for clarity
        self.outc = nn.Conv2d(64, classes, 1)

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return _resize(self.outc(x), size)


__all__ = ["UNet"]
