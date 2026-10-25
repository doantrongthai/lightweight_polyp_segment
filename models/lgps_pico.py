from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from . import register_model

_MIN_TOTAL_PARAMS = 100_000
_MAX_TOTAL_PARAMS = 200_000


@dataclass
class LGPSPicoConfig:
    """Hyperparameter configuration for LGPSPico architecture and loss."""
    pretrained: bool = True
    decoder_channels: int = 14
    rfb_branch_channels: int = 4
    ra_hidden_channels: int = 10
    bpm_hidden_channels: int = 8
    detail_channels: int = 12
    ds_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    boundary_factor: float = 4.0


def _resize(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize a 4D tensor to target spatial size using bilinear interpolation.
    Input: x [B,C,h,w], size (H,W). Output: [B,C,H,W].
    """
    target = tuple(size)
    if x.shape[-2:] == target:
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=False)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize a 4D tensor to match spatial dimensions of a reference tensor.
    Input: x [B,C,h,w], ref [B,D,H,W]. Output: [B,C,H,W].
    """
    return _resize(x, ref.shape[-2:])


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation block.
    Input: [B,Cin,H,W]. Output: [B,Cout,H/s,W/s].
    """
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
    """Depthwise-separable 2D convolution block.
    Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
    """
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
    """Efficient Channel Attention module.
    Input: [B,C,H,W]. Output: [B,C,H,W].
    """
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class ModifiedRFB(nn.Module):
    """Lightweight Modified Receptive Field Block.
    Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
    """
    def __init__(self, in_channels: int, out_channels: int, branch_channels: int = 4) -> None:
        super().__init__()
        self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1, activation=False)
        self.b0 = ConvBNAct(in_channels, branch_channels, kernel_size=1)
        self.b1 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=3, dilation=1),
        )
        self.b2 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=3, dilation=2),
        )
        self.fuse = ConvBNAct(branch_channels * 3, out_channels, kernel_size=1, activation=False)
        self.eca = ECA(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([self.b0(x), self.b1(x), self.b2(x)], dim=1)
        out = self.eca(self.fuse(cat)) + self.shortcut(x)
        return self.act(out)


class LitePPD(nn.Module):
    """Lightweight Parallel Partial Decoder fusing multi-scale features.
    Inputs: f2 [B,16,H/4,W/4], f3 [B,24,H/8,W/8], f4 [B,40,H/16,W/16]. Output: Sg [B,1,H/8,W/8].
    """
    def __init__(
        self,
        in_channels_tuple: Tuple[int, int, int] = (16, 24, 40),
        decoder_channels: int = 14,
        rfb_branch_channels: int = 4,
    ) -> None:
        super().__init__()
        c2, c3, c4 = in_channels_tuple
        dec = decoder_channels

        self.proj2 = nn.Sequential(
            ConvBNAct(c2, dec, kernel_size=1),
            ModifiedRFB(dec, dec, branch_channels=rfb_branch_channels),
        )
        self.proj3 = nn.Sequential(
            ConvBNAct(c3, dec, kernel_size=1),
            ModifiedRFB(dec, dec, branch_channels=rfb_branch_channels),
        )
        self.proj4 = nn.Sequential(
            ConvBNAct(c4, dec, kernel_size=1),
            ModifiedRFB(dec, dec, branch_channels=rfb_branch_channels),
        )

        self.up_conv1 = DepthwiseSeparableConv(dec, dec, kernel_size=3)
        self.up_conv2 = DepthwiseSeparableConv(2 * dec, 2 * dec, kernel_size=3)
        self.concat_conv = DepthwiseSeparableConv(3 * dec, dec, kernel_size=3)
        self.head = nn.Sequential(
            DepthwiseSeparableConv(dec, dec, kernel_size=3),
            ECA(dec),
            nn.Conv2d(dec, 1, kernel_size=1),
        )

    def forward(
        self,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)

        p4_at_3 = _resize_like(p4, p3)
        p2_at_3 = _resize_like(p2, p3)

        x3_1 = self.up_conv1(p4_at_3) * p3
        x3_2 = self.concat_conv(torch.cat([x3_1, p4_at_3, p2_at_3], dim=1))

        sg = self.head(x3_2)
        return p2, p3, p4, sg


class BoundaryPerceptionModule(nn.Module):
    """Boundary Perception Module combining depthwise convolutions and Laplacian prior.
    Input: [B,Cin,H,W]. Output: [B,1,H,W].
    """
    def __init__(self, in_channels: int, hidden_channels: int = 8) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", kernel)

        self.b1 = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=3, dilation=1)
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_map = x.mean(dim=1, keepdim=True)
        lap_edge = torch.abs(F.conv2d(mean_map, self.laplacian, padding=1))
        lap_edge = lap_edge / (lap_edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        learned_gate = self.fuse(self.b1(x))
        return learned_gate * (1.0 + 0.5 * lap_edge)


class ReverseAttention(nn.Module):
    """Reverse Attention block for residual edge refinement.
    Inputs: feature [B,Cin,H,W], prev_pred [B,1,h,w]. Outputs: s_i, r_i, refined.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 10, bpm_hidden: int = 8) -> None:
        super().__init__()
        self.bpm = BoundaryPerceptionModule(in_channels, hidden_channels=bpm_hidden)
        self.pre = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=3)
        self.refine = DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3)
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
        refined = self.eca(self.refine(x * r_i)) + x
        s_i = self.pred(refined) + s_up
        return s_i, r_i, refined


class ShallowDetailRefiner(nn.Module):
    """Lightweight detail injection from stride-2 f1 feature map.
    Inputs: f1 [B,16,H/2,W/2], r3_feat [B,hidden,H/8,W/8]. Output: [B,detail_channels,H/4,W/4].
    """
    def __init__(self, in_hidden: int = 10, detail_channels: int = 12) -> None:
        super().__init__()
        self.p1 = DepthwiseSeparableConv(16, 8, kernel_size=3)
        self.fuse = DepthwiseSeparableConv(in_hidden + 8, detail_channels, kernel_size=3)
        self.eca = ECA(detail_channels)

    def forward(self, f1: torch.Tensor, r3_feat: torch.Tensor) -> torch.Tensor:
        f1_s4 = _resize_like(self.p1(f1), r3_feat)
        r3_s4 = _resize_like(r3_feat, f1_s4)
        return self.eca(self.fuse(torch.cat([r3_s4, f1_s4], dim=1)))


class MobileNetV3SmallPicoBackbone(nn.Module):
    """Shallow ImageNet-pretrained MobileNetV3-Small encoder trunk (features[0:7]).
    Input: RGB image [B,3,H,W]. Outputs: f1 (stride 2), f2 (stride 4), f3 (stride 8), f4 (stride 16).
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v3_small(weights=weights).features

        self.stage1 = features[0:1]   # 16 channels, stride 2
        self.stage2 = features[1:2]   # 16 channels, stride 4
        self.stage3 = features[2:4]   # 24 channels, stride 8
        self.stage4 = features[4:7]   # 40 channels, stride 16
        self.out_channels = (16, 16, 24, 40)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return f1, f2, f3, f4


class WeightedIoUBCELoss(nn.Module):
    """Weighted BCE + Weighted IoU loss with Laplacian boundary enhancement."""
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


@register_model("lgps_pico")
class LGPSPico(nn.Module):
    """Ultra-lightweight polyp segmentation model targeting 100K–200K parameters.
    Input: RGB tensor [B,3,H,W]. Output: single prediction tensor or attention dict.
    """
    def __init__(
        self,
        classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 14,
        rfb_branch_channels: int = 4,
        ra_hidden_channels: int = 10,
        bpm_hidden_channels: int = 8,
        ds_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        boundary_factor: float = 4.0,
        apply_sigmoid: bool = False,
        config: Optional[LGPSPicoConfig] = None,
    ) -> None:
        super().__init__()
        if classes != 1:
            raise ValueError("LGPSPico supports binary polyp segmentation with classes=1.")

        cfg = config or LGPSPicoConfig(
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

        self.encoder = MobileNetV3SmallPicoBackbone(pretrained=cfg.pretrained)
        _, c2, c3, c4 = self.encoder.out_channels
        dec_ch = cfg.decoder_channels
        ra_ch = cfg.ra_hidden_channels
        bpm_ch = cfg.bpm_hidden_channels

        self.ppd = LitePPD(
            in_channels_tuple=(c2, c3, c4),
            decoder_channels=dec_ch,
            rfb_branch_channels=cfg.rfb_branch_channels,
        )
        self.ra3 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)
        self.ra2 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)

        self.detail = ShallowDetailRefiner(in_hidden=ra_ch, detail_channels=cfg.detail_channels)
        self.detail_head = nn.Conv2d(cfg.detail_channels, 1, kernel_size=1)

        self.loss_fn = WeightedIoUBCELoss(boundary_factor=cfg.boundary_factor)
        self.ds_weights = cfg.ds_weights
        self._verify_parameter_budget()

    def _verify_parameter_budget(self) -> None:
        total = self.parameter_count()
        if not (_MIN_TOTAL_PARAMS <= total <= _MAX_TOTAL_PARAMS):
            raise RuntimeError(
                f"LGPSPico parameter budget violation: {total:,} parameters "
                f"(must be between {_MIN_TOTAL_PARAMS:,} and {_MAX_TOTAL_PARAMS:,})."
            )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        size = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)
        p2, p3, p4, sg_raw = self.ppd(f2, f3, f4)

        s3_raw, r3_raw, r3_feat = self.ra3(p3, sg_raw)
        s2_raw, r2_raw, _ = self.ra2(p2, s3_raw)

        detail_feat = self.detail(f1, r3_feat)
        s2_refined = self.detail_head(detail_feat) + _resize_like(s2_raw, detail_feat)

        S2 = _resize(s2_refined, size)
        if not return_attn and not self.training:
            return torch.sigmoid(S2) if self.apply_sigmoid else S2

        S3 = _resize(s3_raw, size)
        Sg = _resize(sg_raw, size)

        w = self.ds_weights
        fused_logits = w[0] * S2 + w[1] * S3 + w[2] * Sg
        pred = torch.sigmoid(fused_logits) if self.apply_sigmoid else fused_logits

        if not return_attn:
            return pred

        return {
            "pred": pred,
            "S3": S3,
            "S2": S2,
            "Sg": Sg,
            "R3": _resize(r3_raw, size),
            "R2": _resize(r2_raw, size),
        }

    def forward_with_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(x, return_attn=True)

    def compute_loss(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.loss_fn(outputs, target)

        heads = [outputs["S2"], outputs["S3"], outputs["Sg"]]
        loss_total = sum(w * self.loss_fn(h, target) for w, h in zip(self.ds_weights, heads))
        return loss_total

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "LGPSPico",
    "LGPSPicoConfig",
    "MobileNetV3SmallPicoBackbone",
    "LitePPD",
    "ReverseAttention",
    "BoundaryPerceptionModule",
]