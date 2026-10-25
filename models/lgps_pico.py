from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from . import register_model

_MIN_TOTAL_PARAMS = 250_000
_MAX_TOTAL_PARAMS = 700_000


@dataclass
class LGPSPicoConfig:
    pretrained: bool = True
    decoder_channels: int = 32
    ra_hidden_channels: int = 24
    bpm_hidden_channels: int = 16
    detail_channels: int = 24
    ds_weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)


def _resize(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    target = tuple(size)
    if x.shape[-2:] == target:
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=False)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return _resize(x, ref.shape[-2:])


class ConvBNAct(nn.Module):
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


class CrossDomainStyleNorm(nn.Module):
    """
    Khối chuẩn hóa thích ứng miền (Domain-Adaptive Style Normalization):
    Kết hợp InstanceNorm và BatchNorm giúp xóa bỏ sự lệch tông màu nội soi
    giữa Kvasir/ClinicDB (train) và ColonDB/ETIS (test).
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_norm = nn.InstanceNorm2d(channels, affine=True)
        self.bn_norm = nn.BatchNorm2d(channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.alpha)
        return w * self.in_norm(x) + (1.0 - w) * self.bn_norm(x)


class CoordinateSpatialAttention(nn.Module):
    """
    Chú ý tọa độ không gian dọc - ngang (Coordinate Strip Attention):
    Giúp mạng định vị chính xác các polyp kích thước siêu nhỏ trong ETIS
    và polyp dẹt nằm dọc nếp gấp ruột trong ColonDB.
    """
    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = ConvBNAct(channels, mid, kernel_size=1)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class ProgressiveFeatureAggregation(nn.Module):
    """
    Mô-đun tổng hợp đặc trưng đa tỷ lệ lũy tiến (lấy cảm hứng từ Lite-PolypNet):
    Sử dụng 4 nhánh giãn nở (dilation 1, 2, 4, 6) nối tiếp nhau để mở rộng
    trường nhìn mà vẫn giữ nguyên độ phân giải cho polyp nhỏ.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.b1 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, dilation=1)
        self.b2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, dilation=2)
        self.b3 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, dilation=4)
        self.b4 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, dilation=6)
        self.fuse = ConvBNAct(out_channels * 4, out_channels, kernel_size=1, activation=False)
        self.coord_attn = CoordinateSpatialAttention(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.proj(x)
        y1 = self.b1(x0)
        y2 = self.b2(x0 + y1)
        y3 = self.b3(x0 + y2)
        y4 = self.b4(x0 + y3)
        out = self.fuse(torch.cat([y1, y2, y3, y4], dim=1))
        out = self.coord_attn(out)
        return self.act(out + x0)


class AdaptivePPD(nn.Module):
    """
    Bộ giải mã ngữ cảnh toàn cục kết hợp Progressive Feature Aggregation.
    """
    def __init__(
        self,
        in_channels_tuple: Tuple[int, int, int] = (24, 48, 96),
        decoder_channels: int = 32,
    ) -> None:
        super().__init__()
        c3, c4, c5 = in_channels_tuple
        dec = decoder_channels

        self.pfa3 = ProgressiveFeatureAggregation(c3, dec)
        self.pfa4 = ProgressiveFeatureAggregation(c4, dec)
        self.pfa5 = ProgressiveFeatureAggregation(c5, dec)

        self.up_conv4 = DepthwiseSeparableConv(dec, dec, kernel_size=3)
        self.up_conv3 = DepthwiseSeparableConv(dec, dec, kernel_size=3)
        self.concat_conv = DepthwiseSeparableConv(3 * dec, dec, kernel_size=3)

        self.head = nn.Sequential(
            DepthwiseSeparableConv(dec, dec, kernel_size=3),
            CoordinateSpatialAttention(dec),
            nn.Conv2d(dec, 1, kernel_size=1),
        )

    def forward(self, f3: torch.Tensor, f4: torch.Tensor, f5: torch.Tensor):
        p3 = self.pfa3(f3)  # [B, dec, 44, 44]
        p4 = self.pfa4(f4)  # [B, dec, 22, 22]
        p5 = self.pfa5(f5)  # [B, dec, 11, 11]

        p5_at_4 = _resize_like(p5, p4)  # [B, dec, 22, 22]
        p5_at_3 = _resize_like(p5, p3)  # [B, dec, 44, 44]
        p4_at_3 = _resize_like(p4, p3)  # [B, dec, 44, 44]

        x4_1 = self.up_conv4(p5_at_4) * p4
        x4_1_at_3 = _resize_like(x4_1, p3)

        x3_1 = self.up_conv3(p5_at_3) * x4_1_at_3 * p3
        x3_2 = self.concat_conv(torch.cat([x3_1, p4_at_3, p5_at_3], dim=1))
        sg = self.head(x3_2)
        return p3, p4, p5, sg


class UncertaintyBoundaryGate(nn.Module):
    """
    Cổng khử nhiễu vùng biên bất định (Uncertainty & Boundary Perception):
    Khuếch đại các vùng có xác suất dự đoán gần 0.5 (vùng ranh giới nhập nhằng ở ColonDB/ETIS)
    kết hợp với đạo hàm bậc hai Laplacian.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", kernel)
        self.b1 = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=3)
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, prev_prob: torch.Tensor) -> torch.Tensor:
        mean_map = x.mean(dim=1, keepdim=True)
        lap_edge = torch.abs(F.conv2d(mean_map, self.laplacian, padding=1))
        lap_edge = lap_edge / (lap_edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        # Bản đồ bất định: đạt cực đại (=1.0) tại ranh giới mờ nơi xác suất = 0.5
        uncertainty = 1.0 - torch.abs(2.0 * prev_prob - 1.0)
        learned_gate = self.fuse(self.b1(x))
        return learned_gate * (1.0 + 0.5 * lap_edge + 0.5 * uncertainty)


class ReverseAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 24, bpm_hidden: int = 16) -> None:
        super().__init__()
        self.ubg = UncertaintyBoundaryGate(in_channels, hidden_channels=bpm_hidden)
        self.pre = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=3)
        self.refine = DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size=3, dilation=2)
        self.attn = CoordinateSpatialAttention(hidden_channels)
        self.pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor, prev_pred: torch.Tensor):
        s_up = _resize_like(prev_pred, feature)
        prob = torch.sigmoid(s_up)
        a_i = 1.0 - prob
        gate = self.ubg(feature, prob)
        r_i = a_i * gate

        x = self.pre(feature)
        refined = self.attn(self.refine(x * r_i)) + x
        s_i = self.pred(refined) + s_up
        return s_i, r_i, refined


class ShallowDetailRefiner(nn.Module):
    """
    Bộ tinh chỉnh chi tiết tầng nông có tích hợp CrossDomainStyleNorm
    để lọc nhiễu màu sắc trước khi ghép vào nhánh dự đoán cuối.
    """
    def __init__(self, in_hidden: int = 24, detail_channels: int = 24) -> None:
        super().__init__()
        self.norm1 = CrossDomainStyleNorm(16)
        self.norm2 = CrossDomainStyleNorm(16)
        self.p1 = DepthwiseSeparableConv(16, 12, kernel_size=3)
        self.p2 = DepthwiseSeparableConv(16, 12, kernel_size=3)
        self.fuse = DepthwiseSeparableConv(in_hidden + 24, detail_channels, kernel_size=3)
        self.attn = CoordinateSpatialAttention(detail_channels)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, r3_feat: torch.Tensor):
        f1_clean = self.p1(self.norm1(f1))  # [B, 12, 176, 176]
        f2_clean = self.p2(self.norm2(f2))  # [B, 12, 88, 88]

        f1_s4 = _resize_like(f1_clean, f2_clean)   # [B, 12, 88, 88]
        r3_s4 = _resize_like(r3_feat, f2_clean)    # [B, in_hidden, 88, 88]
        return self.attn(self.fuse(torch.cat([r3_s4, f2_clean, f1_s4], dim=1)))


class MobileNetV3SmallMidBackbone(nn.Module):
    """
    Chỉ trích xuất features[0:11] của MobileNetV3-Small (~422K params),
    bỏ qua block 11 và 12 siêu nặng để giữ tổng tham số toàn mạng ~512K (< 700K).
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v3_small(weights=weights).features

        self.stage1 = features[0:1]   # 16 ch, stride 2  (176x176)
        self.stage2 = features[1:2]   # 16 ch, stride 4  (88x88)
        self.stage3 = features[2:4]   # 24 ch, stride 8  (44x44)
        self.stage4 = features[4:9]   # 48 ch, stride 16 (22x22)
        self.stage5 = features[9:11]  # 96 ch, stride 32 (11x11)
        self.out_channels = (16, 16, 24, 48, 96)

    def forward(self, x: torch.Tensor):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return f1, f2, f3, f4, f5


@register_model("lgps_pico")
class LGPSPico(nn.Module):
    """
    Kiến trúc LGPSPico (~512K params):
    - Tích hợp CrossDomainStyleNorm chống lệch miền trên ColonDB & ETIS.
    - Tích hợp ProgressiveFeatureAggregation (dilation 1, 2, 4, 6) + CoordinateSpatialAttention.
    - Tích hợp UncertaintyBoundaryGate trong Reverse Attention.
    """
    def __init__(
        self,
        classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 32,
        ra_hidden_channels: int = 24,
        bpm_hidden_channels: int = 16,
        ds_weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
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
        )
        self.config = cfg
        self.apply_sigmoid = apply_sigmoid

        self.encoder = MobileNetV3SmallMidBackbone(pretrained=cfg.pretrained)
        _, _, c3, c4, c5 = self.encoder.out_channels
        dec_ch = cfg.decoder_channels
        ra_ch = cfg.ra_hidden_channels
        bpm_ch = cfg.bpm_hidden_channels

        self.ppd = AdaptivePPD(in_channels_tuple=(c3, c4, c5), decoder_channels=dec_ch)
        self.ra5 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)
        self.ra4 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)
        self.ra3 = ReverseAttention(dec_ch, hidden_channels=ra_ch, bpm_hidden=bpm_ch)

        self.detail = ShallowDetailRefiner(in_hidden=ra_ch, detail_channels=cfg.detail_channels)
        self.detail_head = nn.Conv2d(cfg.detail_channels, 1, kernel_size=1)

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
        f1, f2, f3, f4, f5 = self.encoder(x)
        p3, p4, p5, sg_raw = self.ppd(f3, f4, f5)

        s5_raw, r5_raw, _ = self.ra5(p5, sg_raw)
        s4_raw, r4_raw, _ = self.ra4(p4, s5_raw)
        s3_raw, r3_raw, r3_feat = self.ra3(p3, s4_raw)

        # Tinh chỉnh ở độ phân giải cao (stride 4 = 88x88)
        detail_feat = self.detail(f1, f2, r3_feat)
        s_final_raw = self.detail_head(detail_feat) + _resize_like(s3_raw, detail_feat)

        S3 = _resize(s_final_raw, size)
        S4 = _resize(s4_raw, size)
        S5 = _resize(s5_raw, size)
        Sg = _resize(sg_raw, size)

        # Kết hợp đa tỷ lệ cả khi train lẫn eval để đồng nhất phân phối dự đoán
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

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "LGPSPico",
    "LGPSPicoConfig",
]
