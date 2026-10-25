import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from . import register_model

_MAX_TOTAL_PARAMS = 1_000_000


@dataclass
class LGPSTinyConfig:
    """Hyperparameter configuration for LGPSTiny architecture."""
    pretrained: bool = True
    truncate_576: bool = True
    decoder_channels: int = 24
    rfb_branch_channels: int = 8
    ra_hidden_channels: int = 16
    bpm_hidden_channels: int = 16
    detail_channels: int = 16
    ds_weights: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.1)
    boundary_factor: float = 4.0


def _resize(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize a 4D tensor to target spatial size using bilinear interpolation."""
    target = tuple(size)
    if x.shape[-2:] == target:
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=True)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize a 4D tensor to match spatial dimensions of a reference tensor."""
    return _resize(x, ref.shape[-2:])


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable 2D convolution block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.dw = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=in_channels,
            activation=True,
        )
        self.pw = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ECA(nn.Module):
    """Efficient Channel Attention module."""
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class ModifiedRFB(nn.Module):
    """Lightweight Modified Receptive Field Block with depthwise-separable branches."""
    def __init__(self, in_channels: int, out_channels: int, branch_channels: int = 8) -> None:
        super().__init__()
        self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1, activation=False)
        self.b0 = ConvBNAct(in_channels, branch_channels, kernel_size=1)
        self.b1 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=3, dilation=1),
        )
        self.b2 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=3, dilation=1),
            DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=3, dilation=3),
        )
        self.b3 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=5, dilation=1),
            DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=3, dilation=5),
        )
        self.fuse = ConvBNAct(branch_channels * 4, out_channels, kernel_size=1, activation=False)
        self.eca = ECA(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.eca(self.fuse(cat)) + self.shortcut(x)
        return self.act(out)


class LitePPD(nn.Module):
    """Lightweight Parallel Partial Decoder."""
    def __init__(
        self,
        in_channels_tuple: Tuple[int, int, int] = (24, 48, 96),
        decoder_channels: int = 24,
        rfb_branch_channels: int = 8,
    ) -> None:
        super().__init__()
        c3, c4, c5 = in_channels_tuple
        dec = decoder_channels

        self.proj3 = nn.Sequential(
            ConvBNAct(c3, dec, kernel_size=1),
            ModifiedRFB(dec, dec, branch_channels=rfb_branch_channels),
        )
        self.proj4 = nn.Sequential(
            ConvBNAct(c4, dec, kernel_size=1),
            ModifiedRFB(dec, dec, branch_channels=rfb_branch_channels),
        )
        self.proj5 = nn.Sequential(
            ConvBNAct(c5, dec, kernel_size=1),
            ModifiedRFB(dec, dec, branch_channels=rfb_branch_channels),
        )

        self.up_conv1 = DepthwiseSeparableConv(dec, dec, kernel_size=3)
        self.up_conv2 = DepthwiseSeparableConv(dec, dec, kernel_size=3)
        self.up_conv3 = DepthwiseSeparableConv(dec, dec, kernel_size=3)
        self.up_conv4 = DepthwiseSeparableConv(dec, dec, kernel_size=3)
        self.up_conv5 = DepthwiseSeparableConv(2 * dec, 2 * dec, kernel_size=3)

        self.concat_conv2 = DepthwiseSeparableConv(2 * dec, 2 * dec, kernel_size=3)
        self.concat_conv3 = DepthwiseSeparableConv(3 * dec, 3 * dec, kernel_size=3)
        self.head = nn.Sequential(
            DepthwiseSeparableConv(3 * dec, dec, kernel_size=3),
            ECA(dec),
            nn.Conv2d(dec, 1, kernel_size=1),
        )

    def forward(
        self,
        f3: torch.Tensor,
        f4: torch.Tensor,
        f5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)
        p5 = self.proj5(f5)

        p5_at_4 = _resize_like(p5, p4)
        p5_at_3 = _resize_like(p5, p3)
        p4_at_3 = _resize_like(p4, p3)

        x4_1 = self.up_conv1(p5_at_4) * p4
        x3_1 = self.up_conv2(p5_at_3) * self.up_conv3(p4_at_3) * p3

        x4_2 = self.concat_conv2(torch.cat([x4_1, self.up_conv4(p5_at_4)], dim=1))
        x4_2_at_3 = _resize_like(x4_2, p3)
        x3_2 = self.concat_conv3(torch.cat([x3_1, self.up_conv5(x4_2_at_3)], dim=1))

        sg = self.head(x3_2)
        return p3, p4, p5, sg


class BoundaryPerceptionModule(nn.Module):
    """Boundary Perception Module (BPM)."""
    def __init__(self, in_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", kernel)

        self.b1 = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=3, dilation=1)
        self.b2 = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=3, dilation=2)
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(hidden_channels * 2, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_map = x.mean(dim=1, keepdim=True)
        lap_edge = torch.abs(F.conv2d(mean_map, self.laplacian, padding=1))
        lap_edge = lap_edge / (lap_edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        learned_gate = self.fuse(torch.cat([self.b1(x), self.b2(x)], dim=1))
        return learned_gate * (1.0 + 0.5 * lap_edge)


class ReverseAttention(nn.Module):
    """Reverse Attention block."""
    def __init__(self, in_channels: int, hidden_channels: int = 16, bpm_hidden: int = 16) -> None:
        super().__init__()
        self.bpm = BoundaryPerceptionModule(in_channels, hidden_channels=bpm_hidden)
        self.pre = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=3)
        self.refine1 = DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3)
        self.refine2 = DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3)
        self.eca = ECA(hidden_channels)
        self.pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(
        self,
        feature: torch.Tensor,
        prev_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_up = _resize_like(prev_pred, feature)
        a_i = 1.0 - torch.sigmoid(s_up)
        bpm_gate = self.bpm(feature)
        r_i = a_i * bpm_gate

        x = self.pre(feature)
        refined = self.eca(self.refine2(self.refine1(x * r_i))) + x
        s_i = self.pred(refined) + s_up
        return s_i, r_i, refined


class ShallowDetailRefiner(nn.Module):
    """Lightweight detail injection from shallow features."""
    def __init__(self, in_hidden: int = 16, detail_channels: int = 16) -> None:
        super().__init__()
        self.p1 = DepthwiseSeparableConv(16, 8, kernel_size=3)
        self.p2 = DepthwiseSeparableConv(16, 8, kernel_size=3)
        self.fuse = DepthwiseSeparableConv(in_hidden + 16, detail_channels, kernel_size=3)
        self.eca = ECA(detail_channels)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, r3_feat: torch.Tensor) -> torch.Tensor:
        f1_s4 = _resize_like(self.p1(f1), f2)
        f2_s4 = self.p2(f2)
        r3_s4 = _resize_like(r3_feat, f2)
        return self.eca(self.fuse(torch.cat([r3_s4, f2_s4, f1_s4], dim=1)))


class MobileNetV3SmallBackbone(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small encoder."""
    def __init__(self, pretrained: bool = True, truncate_576: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v3_small(weights=weights).features

        self.stage1 = features[0:1]
        self.stage2 = features[1:2]
        self.stage3 = features[2:4]
        self.stage4 = features[4:9]
        if truncate_576:
            self.stage5 = features[9:12]
            self.out_channels = (16, 16, 24, 48, 96)
        else:
            self.stage5 = features[9:13]
            self.out_channels = (16, 16, 24, 48, 576)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return f1, f2, f3, f4, f5


class WeightedIoUBCELoss(nn.Module):
    """Weighted BCE + Weighted IoU loss with explicit Laplacian boundary weight."""
    def __init__(self, boundary_factor: float = 4.0, gamma: float = 2.0, pool_kernel: int = 31) -> None:
        super().__init__()
        self.boundary_factor = boundary_factor
        self.gamma = gamma
        self.pool_kernel = pool_kernel
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", kernel)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()
        logits = _resize_like(logits, target)

        lap = torch.abs(F.conv2d(target, self.laplacian, padding=1))
        local_diff = torch.abs(
            F.avg_pool2d(target, self.pool_kernel, stride=1, padding=self.pool_kernel // 2) - target
        )
        w = 1.0 + self.boundary_factor * lap + self.gamma * local_diff

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        wbce = (w * bce).sum(dim=(1, 2, 3)) / (w.sum(dim=(1, 2, 3)) + 1e-6)

        pred = torch.sigmoid(logits)
        inter = (w * pred * target).sum(dim=(1, 2, 3))
        union = (w * (pred + target - pred * target)).sum(dim=(1, 2, 3))
        wiou = 1.0 - (inter + 1.0) / (union + 1.0)
        return (wbce + wiou).mean()


@register_model("lgps_tiny")
class LGPSTiny(nn.Module):
    """Ultra-lightweight PraNet-inspired polyp segmentation network (< 1M params)."""
    def __init__(
        self,
        classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 24,
        rfb_branch_channels: int = 8,
        ra_hidden_channels: int = 16,
        bpm_hidden_channels: int = 16,
        ds_weights: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.1),
        boundary_factor: float = 4.0,
        apply_sigmoid: bool = False,
        config: Optional[LGPSTinyConfig] = None,
    ) -> None:
        super().__init__()
        if classes != 1:
            raise ValueError("LGPSTiny supports binary polyp segmentation with classes=1.")

        cfg = config or LGPSTinyConfig(
            pretrained=pretrained,
            decoder_channels=decoder_channels,
            rfb_branch_channels=rfb_branch_channels,
            ra_hidden_channels=ra_hidden_channels,
            bpm_hidden_channels=bpm_hidden_channels,
            ds_weights=ds_weights,
            boundary_factor=boundary_factor,
        )
        self.config = cfg
        self.apply_sigmoid = apply_sigmoid

        self.encoder = MobileNetV3SmallBackbone(
            pretrained=cfg.pretrained,
            truncate_576=cfg.truncate_576,
        )
        _, _, c3, c4, c5 = self.encoder.out_channels
        dec_ch = cfg.decoder_channels
        ra_ch = cfg.ra_hidden_channels
        bpm_ch = cfg.bpm_hidden_channels

        self.ppd = LitePPD(
            in_channels_tuple=(c3, c4, c5),
            decoder_channels=dec_ch,
            rfb_branch_channels=cfg.rfb_branch_channels,
        )
        self.ra5 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)
        self.ra4 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)
        self.ra3 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)

        self.detail = ShallowDetailRefiner(in_hidden=ra_ch, detail_channels=cfg.detail_channels)
        self.detail_head = nn.Conv2d(cfg.detail_channels, 1, kernel_size=1)

        self.loss_fn = WeightedIoUBCELoss(boundary_factor=cfg.boundary_factor)
        self.ds_weights = cfg.ds_weights
        self._verify_parameter_budget()

    def _verify_parameter_budget(self) -> None:
        total = self.parameter_count()
        if total >= _MAX_TOTAL_PARAMS:
            raise RuntimeError(f"LGPSTiny exceeds total parameter budget: {total:,} >= {_MAX_TOTAL_PARAMS:,}")

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.encoder(x)
        p3, p4, p5, sg_raw = self.ppd(f3, f4, f5)

        s5_raw, r5_raw, _ = self.ra5(p5, sg_raw)
        s4_raw, r4_raw, _ = self.ra4(p4, s5_raw)
        s3_raw, r3_raw, r3_feat = self.ra3(p3, s4_raw)

        detail_feat = self.detail(f1, f2, r3_feat)
        s3_refined = self.detail_head(detail_feat) + _resize_like(s3_raw, detail_feat)

        S3 = _resize(s3_refined, size)
        if not return_attn and not self.training:
            return torch.sigmoid(S3) if self.apply_sigmoid else S3

        S4 = _resize(s4_raw, size)
        S5 = _resize(s5_raw, size)
        Sg = _resize(sg_raw, size)

        w = self.ds_weights
        fused_logits = w[0] * S3 + w[1] * S4 + w[2] * S5 + w[3] * Sg
        pred = torch.sigmoid(fused_logits) if self.apply_sigmoid else fused_logits

        if not return_attn:
            return pred

        return {
            "pred": pred,
            "S3": S3,
            "S4": S4,
            "S5": S5,
            "Sg": Sg,
            "R3": _resize(r3_raw, size),
            "R4": _resize(r4_raw, size),
            "R5": _resize(r5_raw, size),
        }

    def forward_with_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(x, return_attn=True)

    def compute_loss(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep supervision loss to match repository requirements."""
        if isinstance(outputs, torch.Tensor):
            return self.loss_fn(outputs, target)

        heads = [outputs["S3"], outputs["S4"], outputs["S5"], outputs["Sg"]]
        loss_total = sum(w * self.loss_fn(h, target) for w, h in zip(self.ds_weights, heads))
        return loss_total

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "LGPSTiny",
    "LGPSTinyConfig",
    "MobileNetV3SmallBackbone",
    "LitePPD",
    "ReverseAttention",
    "BoundaryPerceptionModule",
]
