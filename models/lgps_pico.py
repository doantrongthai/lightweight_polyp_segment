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
    pretrained: bool = True
    decoder_channels: int = 16
    ra_hidden_channels: int = 12
    bpm_hidden_channels: int = 10
    detail_channels: int = 14
    ds_weights: Tuple[float, float, float, float] = (0.35, 0.35, 0.2, 0.1)
    boundary_factor: float = 6.0
    alpha: float = 0.3  # Trọng số False Positive cho Focal Tversky
    beta: float = 0.7   # Trọng số False Negative (ép bắt polyp nhỏ ở ColonDB/ETIS)
    gamma: float = 2.0  # Hệ số Focal


def _resize(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    target = tuple(size)
    if x.shape[-2:] == target:
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=False)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return _resize(x, ref.shape[-2:])


class GhostModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, ratio: int = 2) -> None:
        super().__init__()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class ECA(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class LightASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling siêu nhẹ giúp mở rộng vùng tiếp nhận 
    để nhận diện các polyp kích thước lớn hoặc bị che khuất.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid = out_channels // 4
        self.branch1 = GhostModule(in_channels, mid, 1)
        self.branch2 = nn.Sequential(
            GhostModule(in_channels, mid, 1),
            nn.Conv2d(mid, mid, 3, padding=2, dilation=2, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            GhostModule(in_channels, mid, 1),
            nn.Conv2d(mid, mid, 3, padding=4, dilation=4, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.fuse = GhostModule(mid * 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        bp = F.interpolate(self.pool_branch(x), size=size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([b1, b2, b3, bp], dim=1))


class GhostPPD(nn.Module):
    def __init__(self, in_channels_tuple: Tuple[int, int, int] = (16, 24, 48), decoder_channels: int = 16) -> None:
        super().__init__()
        c2, c3, c4 = in_channels_tuple
        dec = decoder_channels

        self.proj2 = GhostModule(c2, dec, kernel_size=1)
        self.proj3 = LightASPP(c3, dec)  # Tích hợp LightASPP tại tầng trung tâm để bắt polyp nhỏ
        self.proj4 = GhostModule(c4, dec, kernel_size=1)

        self.up_conv1 = GhostModule(dec, dec, kernel_size=3)
        self.concat_conv = GhostModule(3 * dec, dec, kernel_size=3)
        self.head = nn.Sequential(
            GhostModule(dec, dec, kernel_size=3),
            ECA(dec),
            nn.Conv2d(dec, 1, kernel_size=1),
        )

    def forward(self, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor):
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
    def __init__(self, in_channels: int, hidden_channels: int = 8) -> None:
        super().__init__()
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian", kernel)
        self.b1 = GhostModule(in_channels, hidden_channels, kernel_size=3)
        self.fuse = nn.Sequential(
            GhostModule(hidden_channels, hidden_channels, kernel_size=3),
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
    def __init__(self, in_channels: int, hidden_channels: int = 12, bpm_hidden: int = 10) -> None:
        super().__init__()
        self.bpm = BoundaryPerceptionModule(in_channels, hidden_channels=bpm_hidden)
        self.pre = GhostModule(in_channels, hidden_channels, kernel_size=3)
        self.refine = GhostModule(hidden_channels, hidden_channels, kernel_size=3)
        self.eca = ECA(hidden_channels)
        self.pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor, prev_pred: torch.Tensor):
        s_up = _resize_like(prev_pred, feature)
        a_i = 1.0 - torch.sigmoid(s_up)
        bpm_gate = self.bpm(feature)
        r_i = a_i * bpm_gate

        x = self.pre(feature)
        refined = self.eca(self.refine(x * r_i)) + x
        s_i = self.pred(refined) + s_up
        return s_i, r_i, refined


class ShallowDetailRefiner(nn.Module):
    def __init__(self, in_hidden: int = 12, detail_channels: int = 14) -> None:
        super().__init__()
        self.p1 = GhostModule(16, 8, kernel_size=3)
        self.fuse = GhostModule(in_hidden + 8, detail_channels, kernel_size=3)
        self.eca = ECA(detail_channels)

    def forward(self, f1: torch.Tensor, r3_feat: torch.Tensor):
        f1_s4 = _resize_like(self.p1(f1), r3_feat)
        r3_s4 = _resize_like(r3_feat, f1_s4)
        return self.eca(self.fuse(torch.cat([r3_s4, f1_s4], dim=1)))


class MobileNetV3SmallPicoBackbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v3_small(weights=weights).features

        self.stage1 = features[0:1]
        self.stage2 = features[1:2]
        self.stage3 = features[2:4]
        self.stage4 = features[4:9]
        self.out_channels = (16, 16, 24, 48)

    def forward(self, x: torch.Tensor):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return f1, f2, f3, f4


class FocalTverskyStructureLoss(nn.Module):
    """
    Hàm mất mát kết hợp Focal Tversky Loss và Laplacian Boundary Loss.
    Tversky Loss với beta cao giúp ép mô hình không bỏ sót polyp nhỏ ở ColonDB và ETIS.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 2.0, boundary_factor: float = 6.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.boundary_factor = boundary_factor
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian", kernel)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()
        logits = _resize_like(logits, target)
        pred = torch.sigmoid(logits)

        # Focal Tversky Loss
        num = (pred * target).sum(dim=(1, 2, 3))
        den = num + self.alpha * ((1 - pred) * target).sum(dim=(1, 2, 3)) + self.beta * (pred * (1 - target)).sum(dim=(1, 2, 3))
        tversky = (num + 1.0) / (den + 1.0)
        focal_tversky = torch.pow((1.0 - tversky), self.gamma).mean()

        # Boundary Weighted BCE
        lap = torch.abs(F.conv2d(target, self.laplacian, padding=1))
        w = 1.0 + self.boundary_factor * lap
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        wbce = (w * bce).sum(dim=(1, 2, 3)) / (w.sum(dim=(1, 2, 3)) + 1e-6)

        return focal_tversky + wbce.mean()


@register_model("lgps_pico")
class LGPSPico(nn.Module):
    """
    Mô hình LGPSPico tối ưu chuyên sâu kéo điểm ColonDB và ETIS lên >= 0.7.
    """
    def __init__(
        self,
        classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 16,
        ra_hidden_channels: int = 12,
        bpm_hidden_channels: int = 10,
        ds_weights: Tuple[float, float, float, float] = (0.35, 0.35, 0.2, 0.1),
        boundary_factor: float = 6.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 2.0,
        apply_sigmoid: bool = False,
        config: Optional[LGPSPicoConfig] = None,
    ) -> None:
        super().__init__()
        if classes != 1:
            raise ValueError("LGPSPico hỗ trợ phân đoạn nhị phân polyp với classes=1.")

        cfg = config or LGPSPicoConfig(
            pretrained=pretrained,
            decoder_channels=decoder_channels,
            ra_hidden_channels=ra_hidden_channels,
            bpm_hidden_channels=bpm_hidden_channels,
            ds_weights=ds_weights,
            boundary_factor=boundary_factor,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.config = cfg
        self.apply_sigmoid = apply_sigmoid

        self.encoder = MobileNetV3SmallPicoBackbone(pretrained=cfg.pretrained)
        _, c2, c3, c4 = self.encoder.out_channels
        dec_ch = cfg.decoder_channels
        ra_ch = cfg.ra_hidden_channels
        bpm_ch = cfg.bpm_hidden_channels

        self.ppd = GhostPPD(in_channels_tuple=(c2, c3, c4), decoder_channels=dec_ch)
        self.ra3 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)
        self.ra2 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)

        self.detail = ShallowDetailRefiner(in_hidden=ra_ch, detail_channels=cfg.detail_channels)
        self.detail_head = nn.Conv2d(cfg.detail_channels, 1, kernel_size=1)

        self.edge_head = nn.Sequential(
            GhostModule(dec_ch, 8, kernel_size=3),
            nn.Conv2d(8, 1, kernel_size=1),
        )

        self.loss_fn = FocalTverskyStructureLoss(
            alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma, boundary_factor=cfg.boundary_factor
        )
        self.edge_loss_fn = nn.BCEWithLogitsLoss()
        self.ds_weights = cfg.ds_weights
        self._verify_parameter_budget()

    def _verify_parameter_budget(self) -> None:
        total = self.parameter_count()
        if not (_MIN_TOTAL_PARAMS <= total <= _MAX_TOTAL_PARAMS):
            raise RuntimeError(
                f"Vi phạm ngân sách tham số LGPSPico: {total:,} tham số "
                f"(phải nằm trong khoảng từ {_MIN_TOTAL_PARAMS:,} đến {_MAX_TOTAL_PARAMS:,})."
            )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
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
        Edge = _resize(self.edge_head(p3), size)

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
            "Edge": Edge,
            "R3": _resize(r3_raw, size),
            "R2": _resize(r2_raw, size),
        }

    def forward_with_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(x, return_attn=True)

    def compute_loss(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], target: torch.Tensor):
        if isinstance(outputs, torch.Tensor):
            return self.loss_fn(outputs, target)

        if target.ndim == 3:
            target_mask = target.unsqueeze(1).float()
        else:
            target_mask = target.float()
        
        lap_kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        edge_gt = (torch.abs(F.conv2d(target_mask, lap_kernel, padding=1)) > 0.1).float()

        heads = [outputs["S2"], outputs["S3"], outputs["Sg"]]
        w = self.ds_weights
        loss_seg = sum(w[i] * self.loss_fn(heads[i], target) for i in range(3))
        loss_edge = w[3] * self.edge_loss_fn(outputs["Edge"], edge_gt)

        return loss_seg + loss_edge

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "LGPSPico",
    "LGPSPicoConfig",
    "MobileNetV3SmallPicoBackbone",
    "GhostPPD",
    "ReverseAttention",
    "BoundaryPerceptionModule",
    "FocalTverskyStructureLoss",
]
