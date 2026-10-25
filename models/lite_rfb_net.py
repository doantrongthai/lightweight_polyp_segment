import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.checkpoint import checkpoint

try:
    from . import register_model
except ImportError:
    def register_model(name):
        def deco(cls):
            return cls
        return deco


def resize_like(x, ref):
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class LiteReSEBlock(nn.Module):
    """Lightweight residual spatial + channel refinement."""
    def __init__(self, channels, expansion=1.25, dilation=1):
        super().__init__()
        hidden = max(int(channels * expansion), channels)

        self.pw1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        self.dw = nn.Conv2d(
            hidden, hidden, 3,
            padding=dilation, dilation=dilation,
            groups=hidden, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden)

        self.pw2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.se = SEBlock(channels)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.bn1(self.pw1(x)))
        y = self.act(self.bn2(self.dw(y)))
        y = self.bn3(self.pw2(y))
        y = self.se(y)
        return self.act(x + y)


class LiteMultiScaleContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        branch = max(channels // 4, 12)

        self.pre = ConvBNAct(channels, channels, 1)

        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, branch, 1, bias=False),
            nn.BatchNorm2d(branch),
            nn.GELU(),
        )

        self.dilated = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, branch, 1, bias=False),
            nn.BatchNorm2d(branch),
            nn.GELU(),
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch, 1, bias=False),
            nn.GELU(),
        )

        self.fuse = ConvBNAct(branch * 3, channels, 1)
        self.se = SEBlock(channels)
        self.gamma = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        z = self.pre(x)
        p = self.pool(z)
        p = F.interpolate(p, size=z.shape[-2:], mode="nearest")

        y = torch.cat([self.local(z), self.dilated(z), p], dim=1)
        y = self.se(self.fuse(y))
        return x + self.gamma * y


class EfficientAxialGateLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.h = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.w = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.proj = ConvBNAct(channels, channels, 1, act=False)
        self.gamma = nn.Parameter(torch.tensor(0.15))

    def forward(self, x):
        hg = torch.sigmoid(self.h(x.mean(dim=3))).unsqueeze(3)
        wg = torch.sigmoid(self.w(x.mean(dim=2))).unsqueeze(2)
        y = x * (0.5 * hg + 0.5 * wg)
        return x + self.gamma * self.proj(y)


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, blocks=1):
        super().__init__()
        layers = [ConvBNAct(in_ch, out_ch, 3, 2)]
        for i in range(blocks):
            layers.append(LiteReSEBlock(out_ch, expansion=1.25, dilation=1 if i == 0 else 2))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class BoundaryGateLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.edge(x)


class GatedSkipFusionLite(nn.Module):
    def __init__(self, in_dec, in_skip, out_ch):
        super().__init__()
        self.dec = ConvBNAct(in_dec, out_ch, 1)
        self.skip = ConvBNAct(in_skip, out_ch, 1)
        hidden = max(out_ch // 2, 6)

        self.gate = nn.Sequential(
            nn.Conv2d(out_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1),
            nn.Sigmoid(),
        )
        self.local = ConvBNAct(out_ch, out_ch, 3, groups=out_ch)
        self.mix = ConvBNAct(out_ch, out_ch, 1)

    def forward(self, x, skip):
        x = resize_like(x, skip)
        d = self.dec(x)
        s = self.skip(skip)
        g = self.gate(d + s)
        return self.mix(self.local(d + g * s))


class ReverseAttentionRefineLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, groups=channels),
            ConvBNAct(channels, channels, 1),
        )
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.50))

    def forward(self, x, coarse_logits):
        coarse = torch.sigmoid(F.interpolate(coarse_logits, size=x.shape[-2:], mode="nearest"))
        reverse = 1.0 - coarse
        y = self.refine(x)
        y = y * (0.5 + self.alpha.clamp(0.0, 1.0) * reverse)
        return x + y * self.gate(x)


class DecoderStageLite(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = GatedSkipFusionLite(in_ch, skip_ch, out_ch)
        self.boundary = BoundaryGateLite(out_ch)
        self.refine = LiteReSEBlock(out_ch, expansion=1.25)

    def forward(self, x, skip):
        x = self.fuse(x, skip)
        b = self.boundary(x)
        x = x + x * b
        return self.refine(x)


@register_model("lite_rfb_net")
class LiteRFBNet(nn.Module):
    """
    LiteRFBNet: Optimized lightweight model for polyp segmentation.
    Integrated with framework Registry and GradCAM visualization hooks.
    """
    def __init__(self, classes=1, checkpoint_decoder=True):
        super().__init__()
        self.gradcam_target_layer = None

        c1, c2, c3, c4 = 20, 40, 80, 128
        self.checkpoint_decoder = checkpoint_decoder

        # ---------------- Encoder ----------------
        self.stem = nn.Sequential(
            ConvBNAct(3, c1, 3),
            LiteReSEBlock(c1, expansion=1.25)
        )
        self.enc2 = EncoderStage(c1, c2, 1)
        self.enc3 = EncoderStage(c2, c3, 2)
        self.enc4 = EncoderStage(c3, c4, 2)

        self.context3 = nn.Sequential(
            LiteMultiScaleContext(c3),
            EfficientAxialGateLite(c3)
        )
        self.context4 = nn.Sequential(
            LiteMultiScaleContext(c4),
            EfficientAxialGateLite(c4),
            LiteReSEBlock(c4, expansion=1.25)
        )

        # Định nghĩa tầng trích xuất GradCAM cho framework
        self.gradcam_target_layer = self.context3

        # ---------------- Decoder ----------------
        self.dec3 = DecoderStageLite(c4, c3, c3)
        self.coarse_head = nn.Conv2d(c3, classes, 1)

        self.dec2 = DecoderStageLite(c3, c2, c2)
        self.ra2 = ReverseAttentionRefineLite(c2)

        self.dec1 = DecoderStageLite(c2, c1, c1)
        self.ra1 = ReverseAttentionRefineLite(c1)

        # ---------------- Final Head ----------------
        self.final = nn.Sequential(
            LiteReSEBlock(c1, expansion=1.25),
            ConvBNAct(c1, c1, 3, groups=c1),
            ConvBNAct(c1, c1, 1),
        )
        self.head = nn.Conv2d(c1, classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def _checkpoint(self, fn, *args):
        if self.training and self.checkpoint_decoder and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        size = x.shape[-2:]

        # Encoder forward
        s1 = self.stem(x)
        s2 = self.enc2(s1)
        s3 = self.context3(self.enc3(s2))
        s4 = self.context4(self.enc4(s3))

        # Decoder forward
        d3 = self._checkpoint(self.dec3, s4, s3)
        coarse = self.coarse_head(d3)

        d2 = self._checkpoint(self.dec2, d3, s2)
        d2 = self._checkpoint(self.ra2, d2, coarse)

        d1 = self._checkpoint(self.dec1, d2, s1)
        d1 = self._checkpoint(self.ra1, d1, coarse)

        # Final prediction logits
        logits = self.head(self.final(d1))

        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

        return logits


__all__ = ["LiteRFBNet"]
