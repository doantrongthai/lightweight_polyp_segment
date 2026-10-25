import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


def _resize_like(x, ref, mode="bilinear", align_corners=True):
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode=mode, align_corners=align_corners)


def _max_unpool(x, indices, kernel_size=2, stride=2, output_size=None):
    """
    Wrapper quanh F.max_unpool2d de bypass deterministic constraint.
    max_unpool2d chua duoc PyTorch dang ky la deterministic du thuc te
    ket qua hoan toan deterministic (chi dat gia tri vao dung vi tri indices).
    Dung try/finally de dam bao co global flag luon duoc khoi phuc,
    ke ca khi max_unpool2d nem exception giua chung.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        out = F.max_unpool2d(x, indices, kernel_size=kernel_size, stride=stride, output_size=output_size)
    finally:
        torch.use_deterministic_algorithms(prev)
    return out


class EncoderBlock(nn.Module):
    """Encoder block with conv-bn-relu pattern"""
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class DecoderBlock(nn.Module):
    """Decoder block with conv-bn-relu pattern"""
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


@register_model("segnet")
class SegNet(nn.Module):
    def __init__(self, classes=1, in_channels=3):
        """
        SegNet architecture following VGG16 encoder.
        All blocks are symmetric between encoder and decoder
        per the original paper (Badrinarayanan et al., 2017).

        Args:
            classes: Number of output classes
                     - 1 for binary segmentation (output: 1 channel with sigmoid)
                     - >=2 for multi-class (output: classes channels with softmax)
            in_channels: Number of input channels (default: 3 for RGB)
        """
        super().__init__()

        self.classes = classes

        # Encoder (following VGG16 architecture)
        self.encoder1 = EncoderBlock(in_channels, 64,  num_convs=2)
        self.encoder2 = EncoderBlock(64,  128, num_convs=2)
        self.encoder3 = EncoderBlock(128, 256, num_convs=3)
        self.encoder4 = EncoderBlock(256, 512, num_convs=3)
        self.encoder5 = EncoderBlock(512, 512, num_convs=3)

        # Decoder (symmetric to encoder)
        self.decoder5 = DecoderBlock(512, 512, num_convs=3)

        self.decoder4_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.decoder3_convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        self.decoder2_convs = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )

        # FIX: decoder1 chi con 1 conv (64->64) truoc lop classifier, dung 2 conv
        # tong cong o muc nay (mirror voi encoder1 co 2 conv: conv->conv).
        # Ban cu co them 1 conv thua (3 conv thay vi 2), lam sai so params.
        self.decoder1_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        # FIX: kernel_size=3 (khong phai 1) cho lop classifier cuoi, dung theo
        # paper goc / kien truc VGG16-symmetric chuan (conv 3x3, padding=1).
        self.decoder1_conv2 = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):
        size = x.shape[2:]

        # Encoder
        x1 = self.encoder1(x)
        x1_size = x1.shape[2:]
        x1_pooled, indices1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)

        x2 = self.encoder2(x1_pooled)
        x2_size = x2.shape[2:]
        x2_pooled, indices2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)

        x3 = self.encoder3(x2_pooled)
        x3_size = x3.shape[2:]
        x3_pooled, indices3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)

        x4 = self.encoder4(x3_pooled)
        x4_size = x4.shape[2:]
        x4_pooled, indices4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)

        x5 = self.encoder5(x4_pooled)
        x5_size = x5.shape[2:]
        x5_pooled, indices5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        # Decoder
        # FIX: truyen output_size cho tung lan unpool de tranh crash / sai
        # kich thuoc khi H,W dau vao khong chia het cho 32.
        d5 = _max_unpool(x5_pooled, indices5, output_size=x5_size)
        d5 = self.decoder5(d5)

        d4 = _max_unpool(d5, indices4, output_size=x4_size)
        d4 = self.decoder4_convs(d4)

        d3 = _max_unpool(d4, indices3, output_size=x3_size)
        d3 = self.decoder3_convs(d3)

        d2 = _max_unpool(d3, indices2, output_size=x2_size)
        d2 = self.decoder2_convs(d2)

        d1 = _max_unpool(d2, indices1, output_size=x1_size)
        d1 = self.decoder1_conv1(d1)
        output = self.decoder1_conv2(d1)

        # FIX: an toan cuoi cung neu output_size van lech vi ly do nao do
        return _resize_like(output, x) if output.shape[2:] != size else output


__all__ = ["SegNet"]
