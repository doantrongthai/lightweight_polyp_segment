from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

try:
    from . import register_model
except ImportError:
    def register_model(name: str):
        def decorator(cls):
            return cls
        return decorator


_MAX_TOTAL_PARAMS = 1_000_000


@dataclass
class LGPSTinyConfig:
    """Hyperparameter configuration for LGPSTiny architecture and loss.

    Input: configuration fields. Output: configuration instance.
    Source: Custom configuration for PraNet-inspired lightweight segmentation.
    """

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
    """Resize a 4D tensor to target spatial size using bilinear interpolation.

    Input: x [B,C,h,w], size (H,W). Output: [B,C,H,W].
    Source: Standard FCN/UNet spatial alignment convention.
    """
    target = tuple(size)
    if x.shape[-2:] == target:
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=False)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize a 4D tensor to match the spatial dimensions of a reference tensor.

    Input: x [B,C,h,w], ref [B,D,H,W]. Output: [B,C,H,W].
    Source: PraNet (Fan et al., MICCAI 2020).
    """
    return _resize(x, ref.shape[-2:])


class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation block.

    Input: [B,Cin,H,W]. Output: [B,Cout,H/s,W/s].
    Source: MobileNetV3 / EfficientNet building block (Howard et al., ICCV 2019).
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
        """Apply Conv2d, BatchNorm2d and optional SiLU.

        Input: [B,Cin,H,W]. Output: [B,Cout,H/s,W/s].
        """
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable 2D convolution block (Depthwise kxk + Pointwise 1x1).

    Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
    Source: MobileNets (Howard et al., arXiv 2017).
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
        """Apply depthwise spatial filtering followed by pointwise channel mixing.

        Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
        """
        return self.pw(self.dw(x))


class ECA(nn.Module):
    """Efficient Channel Attention module using a fast 1D convolution.

    Input: [B,C,H,W]. Output: [B,C,H,W].
    Source: ECA-Net (Wang et al., CVPR 2020).
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute and apply channel-wise attention weights.

        Input: [B,C,H,W]. Output: [B,C,H,W].
        """
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class ModifiedRFB(nn.Module):
    """Lightweight Modified Receptive Field Block with depthwise-separable branches.

    Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
    Source: PraNet (Fan et al., MICCAI 2020) and RFBNet (Liu et al., ECCV 2018).
    """

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
        """Aggregate multi-dilation receptive fields with residual shortcut.

        Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
        """
        cat = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.eca(self.fuse(cat)) + self.shortcut(x)
        return self.act(out)


class LitePPD(nn.Module):
    """Lightweight Parallel Partial Decoder using f3, f4, f5 and depthwise-separable convs.

    Inputs: f3 [B,24,H/8,W/8], f4 [B,48,H/16,W/16], f5 [B,96,H/32,W/32].
    Outputs: Tuple of projected features (p3, p4, p5) each [B,C,H_i,W_i] and global map Sg [B,1,H/8,W/8].
    Source: PraNet Parallel Partial Decoder (Fan et al., MICCAI 2020).
    """

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
        """Project f3, f4, f5 to decoder_channels and fuse in parallel into Sg.

        Inputs: f3 [B,24,H/8,W/8], f4 [B,48,H/16,W/16], f5 [B,96,H/32,W/32].
        Outputs: p3 [B,C,H/8,W/8], p4 [B,C,H/16,W/16], p5 [B,C,H/32,W/32], sg [B,1,H/8,W/8].
        """
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
    """Boundary Perception Module (BPM) using depthwise-separable dilated convolutions.

    Input: [B,Cin,H,W]. Output: boundary attention gate [B,1,H,W].
    Source: Boundary-aware polyp segmentation (PraNet / BDG-Net).
    """

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
        """Compute boundary enhancement gate combining learned DS-Conv and Laplacian prior.

        Input: [B,Cin,H,W]. Output: [B,1,H,W].
        """
        mean_map = x.mean(dim=1, keepdim=True)
        lap_edge = torch.abs(F.conv2d(mean_map, self.laplacian, padding=1))
        lap_edge = lap_edge / (lap_edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        learned_gate = self.fuse(torch.cat([self.b1(x), self.b2(x)], dim=1))
        return learned_gate * (1.0 + 0.5 * lap_edge)


class ReverseAttention(nn.Module):
    """Reverse Attention block: A_i = (1 - sigmoid(upsample(S_{i+1}))) * BPM(f_i).

    Inputs: feature [B,Cin,H,W], prev_pred [B,1,h,w].
    Outputs: s_i [B,1,H,W], r_i (attention map) [B,1,H,W], refined_feat [B,hidden,H,W].
    Source: PraNet (Fan et al., MICCAI 2020).
    """

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
        """Mine complementary boundary regions using reverse attention and BPM.

        Inputs: feature [B,Cin,H,W], prev_pred [B,1,h,w].
        Outputs: s_i [B,1,H,W], r_i [B,1,H,W], refined [B,hidden,H,W].
        """
        s_up = _resize_like(prev_pred, feature)
        a_i = 1.0 - torch.sigmoid(s_up)
        bpm_gate = self.bpm(feature)
        r_i = a_i * bpm_gate

        x = self.pre(feature)
        refined = self.eca(self.refine2(self.refine1(x * r_i))) + x
        s_i = self.pred(refined) + s_up
        return s_i, r_i, refined


class ShallowDetailRefiner(nn.Module):
    """Lightweight detail injection from stride-4 f2 and stride-2 f1 into RA3 output.

    Inputs: f1 [B,16,H/2,W/2], f2 [B,16,H/4,W/4], r3_feat [B,hidden,H/8,W/8].
    Output: refined high-resolution feature [B,hidden,H/4,W/4].
    Source: U-Net / CaraNet shallow boundary refinement.
    """

    def __init__(self, in_hidden: int = 16, detail_channels: int = 16) -> None:
        super().__init__()
        self.p1 = DepthwiseSeparableConv(16, 8, kernel_size=3)
        self.p2 = DepthwiseSeparableConv(16, 8, kernel_size=3)
        self.fuse = DepthwiseSeparableConv(in_hidden + 16, detail_channels, kernel_size=3)
        self.eca = ECA(detail_channels)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, r3_feat: torch.Tensor) -> torch.Tensor:
        """Align and fuse shallow edge cues at stride-4 resolution.

        Inputs: f1 [B,16,H/2,W/2], f2 [B,16,H/4,W/4], r3_feat [B,C,H/8,W/8].
        Output: [B,detail_channels,H/4,W/4].
        """
        f1_s4 = _resize_like(self.p1(f1), f2)
        f2_s4 = self.p2(f2)
        r3_s4 = _resize_like(r3_feat, f2)
        return self.eca(self.fuse(torch.cat([r3_s4, f2_s4, f1_s4], dim=1)))


class MobileNetV3SmallBackbone(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small encoder without classifier or 576-ch expansion.

    Input: RGB image [B,3,H,W].
    Outputs:
        f1: [B,16,H/2,W/2]   (features[0])
        f2: [B,16,H/4,W/4]   (features[1])
        f3: [B,24,H/8,W/8]   (features[2..3])
        f4: [B,48,H/16,W/16] (features[4..8] - full stride-16 stage including 48-ch blocks)
        f5: [B,96,H/32,W/32] (features[9..11] - 48->96 stage, cutting features[12] 576-ch)
    Source: Howard et al., "Searching for MobileNetV3", ICCV 2019.
    """

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
        """Extract 5 hierarchical feature maps at strides 2, 4, 8, 16, 32.

        Input: [B,3,H,W]. Outputs: (f1, f2, f3, f4, f5).
        """
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return f1, f2, f3, f4, f5


@register_model("lgps_tiny")
class LGPSTiny(nn.Module):
    """Ultra-lightweight PraNet-inspired polyp segmentation network (< 1M params, < 0.6 GFLOPs).

    Input: RGB tensor [B,3,H,W].
    Output:
        - Normal mode (return_attn=False): single logits tensor [B,1,H,W] (or probabilities if
          apply_sigmoid=True), seamlessly compatible with standard repository evaluation loops
          while fusing gradients across (S3, S4, S5, Sg) during single-output training.
        - Attention mode (return_attn=True or forward_with_attention): dictionary containing
          {"pred", "S3", "S4", "S5", "Sg", "R3", "R4", "R5"}.
    Source: Inspired by PraNet (Fan et al., MICCAI 2020) and MobileNetV3 (Howard et al., ICCV 2019).
    """

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

        self.register_buffer(
            "ds_weights_buf",
            torch.tensor(cfg.ds_weights, dtype=torch.float32),
        )
        self._verify_parameter_budget()

    def _verify_parameter_budget(self) -> None:
        """Assert that total parameters (including non-trainable) are strictly < 1,000,000."""
        total = self.parameter_count(trainable_only=False)
        if total >= _MAX_TOTAL_PARAMS:
            raise RuntimeError(
                f"LGPSTiny exceeds total parameter budget: {total:,} >= {_MAX_TOTAL_PARAMS:,}"
            )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute forward pass.

        Input: x [B,3,H,W], return_attn (bool).
        Output: Tensor [B,1,H,W] when return_attn=False; Dict[str, Tensor] when return_attn=True.
        """
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

        w = self.ds_weights_buf
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
        """Return predictions along with Sg and Reverse Attention maps R3, R4, R5.

        Input: x [B,3,H,W]. Output: Dict with keys pred, S3, S4, S5, Sg, R3, R4, R5.
        """
        return self.forward(x, return_attn=True)

    def parameter_count(self, trainable_only: bool = False) -> int:
        """Count model parameters.

        Input: trainable_only (bool). Output: exact integer count of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_report(self) -> Dict[str, float]:
        """Return summary dictionary of parameter counts against the 1M budget."""
        total = self.parameter_count(trainable_only=False)
        trainable = self.parameter_count(trainable_only=True)
        return {
            "total_params": float(total),
            "trainable_params": float(trainable),
            "params_m": float(total / 1e6),
            "under_1m": bool(total < _MAX_TOTAL_PARAMS),
        }


__all__ = [
    "LGPSTiny",
    "LGPSTinyConfig",
    "MobileNetV3SmallBackbone",
    "LitePPD",
    "ReverseAttention",
    "BoundaryPerceptionModule",
]
