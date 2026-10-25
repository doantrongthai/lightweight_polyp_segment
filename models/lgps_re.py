from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from . import register_model


_MAX_TRAINABLE_PARAMS = 1_000_000


def _resize(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize a feature tensor with bilinear interpolation.

    Input: x [B,C,h,w]. Output: [B,C,H,W]. This follows the resizing convention
    of the repository UNet skeleton and is used to align multiscale features.
    """
    target = tuple(size)
    if x.shape[-2:] == target:
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=True)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize x to the spatial resolution of ref.

    Input: x [B,C,h,w], ref [B,D,H,W]. Output: [B,C,H,W].
    """
    return _resize(x, ref.shape[-2:])


class ConvBNAct(nn.Module):
    """Compact convolutional projection block.

    Input: [B,Cin,H,W]. Output: [B,Cout,H/s,W/s]. BatchNorm and SiLU provide
    lightweight nonlinear feature transformation without dense decoder blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1, activation: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization and activation.

        Input: [B,Cin,H,W]. Output: [B,Cout,H/s,W/s].
        """
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable 3x3 plus pointwise 1x1 convolution.

    Input: [B,Cin,H,W]. Output: [B,Cout,H,W]. This replaces dense 3x3 decoder
    convolutions with the lightweight factorization commonly used by MobileNet.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: bool = True) -> None:
        super().__init__()
        self.dw = ConvBNAct(in_channels, in_channels, 3, groups=in_channels)
        self.pw = ConvBNAct(in_channels, out_channels, 1, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise and pointwise filtering.

        Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
        """
        return self.pw(self.dw(x))


class ECA(nn.Module):
    """Efficient Channel Attention with parameter-efficient local channel mixing.

    Input: [B,C,H,W]. Output: [B,C,H,W]. ECA avoids the parameter-heavy two-layer
    squeeze-excitation MLP and is used to recover discriminative channel cues.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ECA channel reweighting.

        Input: [B,C,H,W]. Output: [B,C,H,W].
        """
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class SpatialAttention(nn.Module):
    """Compact CBAM-style spatial attention.

    Input: [B,C,H,W]. Output: [B,1,H,W]. Average and maximum channel pooling
    provide a low-parameter spatial descriptor for polyp region localization.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a spatial attention map.

        Input: [B,C,H,W]. Output: [B,1,H,W] in [0,1].
        """
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class BoundaryPriority(nn.Module):
    """Parameter-free Laplacian boundary prior followed by learnable gating.

    Input: [B,C,H,W]. Output: [B,1,H,W]. The fixed 3x3 Laplacian highlights
    local feature transitions, while spatial attention learns a soft boundary
    confidence used by reverse attention.
    """

    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", kernel)
        self.spatial = SpatialAttention(5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the boundary-priority gate.

        Input: [B,C,H,W]. Output: [B,1,H,W] in approximately [0.5,1].
        """
        base = x.mean(dim=1, keepdim=True)
        edge = torch.abs(F.conv2d(base, self.laplacian, padding=1))
        edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        learned = self.spatial(x)
        return 0.5 + 0.25 * edge + 0.25 * learned


class LiteRFB(nn.Module):
    """Three-branch receptive-field fusion using depthwise spatial filters.

    Input: [B,Cin,H,W]. Output: [B,Cout,H,W]. Dilations 1, 3 and 5 aggregate
    local and broader context while avoiding dense convolutional branches.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch = max(out_channels // 4, 8)
        self.b1 = DepthwiseSeparableConv(in_channels, branch)
        self.b2 = nn.Sequential(
            ConvBNAct(in_channels, branch, 1),
            nn.Conv2d(branch, branch, 3, padding=3, dilation=3, groups=branch, bias=False),
            nn.BatchNorm2d(branch),
            nn.SiLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            ConvBNAct(in_channels, branch, 1),
            nn.Conv2d(branch, branch, 3, padding=5, dilation=5, groups=branch, bias=False),
            nn.BatchNorm2d(branch),
            nn.SiLU(inplace=True),
        )
        self.fuse = ConvBNAct(branch * 3, out_channels, 1)
        self.eca = ECA(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate multiscale receptive fields.

        Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
        """
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.eca(self.fuse(y))


class LitePPD(nn.Module):
    """Lightweight Parallel Partial Decoder inspired by PraNet PPD.

    Inputs: f3 [B,24,H3,W3], f4 [B,40,H4,W4], f5 [B,96,H5,W5]. Output:
    global feature [B,32,H3,W3]. Each branch is projected independently, aligned
    to f3, then fused with depthwise-separable and receptive-field operations.
    """

    def __init__(self, channels: Tuple[int, int, int] = (24, 40, 96), width: int = 32) -> None:
        super().__init__()
        self.p3 = ConvBNAct(channels[0], width, 1)
        self.p4 = ConvBNAct(channels[1], width, 1)
        self.p5 = ConvBNAct(channels[2], width, 1)
        self.mix = DepthwiseSeparableConv(width * 3, width)
        self.rfb = LiteRFB(width, width)
        self.eca = ECA(width)

    def forward(self, f3: torch.Tensor, f4: torch.Tensor, f5: torch.Tensor) -> torch.Tensor:
        """Fuse three partial feature levels in parallel.

        Inputs: f3 [B,24,H3,W3], f4 [B,40,H4,W4], f5 [B,96,H5,W5]. Output:
        [B,32,H3,W3].
        """
        p3 = self.p3(f3)
        p4 = _resize_like(self.p4(f4), f3)
        p5 = _resize_like(self.p5(f5), f3)
        y = self.mix(torch.cat([p3, p4, p5], dim=1))
        return self.eca(self.rfb(y) + y)


class ReverseAttention(nn.Module):
    """PraNet-style reverse attention with spatial and boundary refinement.

    Input: feature Fi [B,C,H,W] and coarse prediction S [B,1,h,w]. Outputs:
    refined logits Si [B,1,H,W], reverse-boundary attention Ai [B,1,H,W], and
    refined feature Ri [B,hidden,H,W]. The reverse weight follows Ai=1-sigmoid(S)
    before multiplication by the boundary-priority gate.
    """

    def __init__(self, in_channels: int, hidden: int = 24) -> None:
        super().__init__()
        self.pre = DepthwiseSeparableConv(in_channels, hidden)
        self.eca = ECA(hidden)
        self.spatial = SpatialAttention(5)
        self.boundary = BoundaryPriority()
        self.refine = DepthwiseSeparableConv(hidden, hidden)
        self.pred = nn.Conv2d(hidden, 1, 3, padding=1, bias=True)

    def forward(self, feature: torch.Tensor, previous_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine the complementary region left by the coarse prediction.

        Inputs: feature [B,C,H,W], previous_prediction [B,1,h,w]. Outputs:
        prediction [B,1,H,W], attention [B,1,H,W], refined feature [B,hidden,H,W].
        """
        x = self.pre(feature)
        x = self.eca(x)
        x = x * self.spatial(x)
        previous = _resize_like(previous_prediction, x)
        ai = 1.0 - torch.sigmoid(previous)
        bpm = self.boundary(x)
        attention = ai * bpm
        refined = self.refine(x * attention) + x
        prediction = self.pred(refined)
        return prediction, attention, refined


class LowLevelBoundaryFusion(nn.Module):
    """Low-level detail branch for recovering fine boundary structure.

    Inputs: f2 [B,16,H2,W2], r3 [B,24,H3,W3]. Output: [B,24,H3,W3]. The f2
    feature is projected with a depthwise-separable block and fused only at the
    final refinement stage to limit memory and parameter overhead.
    """

    def __init__(self, low_channels: int = 16, refined_channels: int = 24) -> None:
        super().__init__()
        self.low = DepthwiseSeparableConv(low_channels, 8)
        self.fuse = ConvBNAct(8 + refined_channels, refined_channels, 1)
        self.eca = ECA(refined_channels)

    def forward(self, f2: torch.Tensor, r3: torch.Tensor) -> torch.Tensor:
        """Fuse low-level detail with the finest reverse-attention feature.

        Inputs: f2 [B,16,H2,W2], r3 [B,24,H3,W3]. Output: [B,24,H3,W3].
        """
        low = _resize_like(self.low(f2), r3)
        return self.eca(self.fuse(torch.cat([r3, low], dim=1)) + r3)


class MobileNetV3SmallTruncated(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small feature trunk under a tiny budget.

    Input: RGB [B,3,H,W]. Outputs f1..f5 with channels (16,16,24,40,96) and
    strides approximately (2,4,8,16,32). Only feature blocks 0..10 are retained;
    the classification head and final feature projection are removed. This keeps
    a substantial pretrained representation while avoiding the full 2.54M-parameter
    classifier model.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.blocks = nn.Sequential(*list(base.features[:11]))
        self.out_channels = (16, 16, 24, 40, 96)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract five hierarchical feature maps from the retained pretrained trunk.

        Input: [B,3,H,W]. Outputs are f1 [B,16,H/2,W/2], f2 [B,16,H/4,W/4],
        f3 [B,24,H/8,W/8], f4 [B,40,H/16,W/16], and f5 [B,96,H/32,W/32].
        """
        y = x
        f1 = self.blocks[0](y)
        f2 = self.blocks[1](f1)
        y = self.blocks[2](f2)
        f3 = self.blocks[3](y)
        y = self.blocks[4](f3)
        y = self.blocks[5](y)
        f4 = self.blocks[6](y)
        y = self.blocks[7](f4)
        y = self.blocks[8](y)
        y = self.blocks[9](y)
        f5 = self.blocks[10](y)
        return f1, f2, f3, f4, f5


class WeightedIoUBCELoss(nn.Module):
    """PraNet-style weighted BCE plus weighted IoU loss with Laplacian emphasis.

    Input: logits [B,1,H,W], target [B,1,H,W]. Output: scalar loss. The local
    average-pooling discrepancy provides the PraNet structural weight and the
    Laplacian factor adds extra emphasis near sharp mask transitions.
    """

    def __init__(self, gamma: float = 5.0, pool_kernel: int = 31,
                 boundary_strength: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.pool_kernel = pool_kernel
        self.boundary_strength = boundary_strength
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", kernel)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE and weighted IoU.

        Input: logits [B,1,H,W], target [B,1,H,W]. Output: scalar tensor.
        """
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()
        logits = _resize_like(logits, target)
        pred = torch.sigmoid(logits)
        p = self.pool_kernel // 2
        local = F.avg_pool2d(target, self.pool_kernel, stride=1, padding=p)
        weight = 1.0 + self.gamma * torch.abs(local - target)
        edge = torch.abs(F.conv2d(target, self.laplacian, padding=1))
        edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        weight = weight * (1.0 + self.boundary_strength * edge)
        wbce = (
            weight * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        ).sum(dim=(1, 2, 3)) / (weight.sum(dim=(1, 2, 3)) + 1e-6)
        inter = (weight * pred * target).sum(dim=(1, 2, 3))
        union = (weight * (pred + target - pred * target)).sum(dim=(1, 2, 3))
        wiou = 1.0 - (inter + 1.0) / (union + 1.0)
        return (wbce + wiou).mean()


class TinyDeepSupervisionLoss(nn.Module):
    """Deep-supervision wrapper for S3, S4, S5 and Sg predictions.

    Inputs: dictionary from forward_with_attention and target [B,1,H,W]. Output:
    scalar loss. The default weights emphasize the finest prediction while
    retaining gradients through all reverse-attention stages and the global map.
    """

    def __init__(self, weights: Sequence[float] = (0.4, 0.3, 0.2, 0.1)) -> None:
        super().__init__()
        if len(weights) != 4:
            raise ValueError("weights must contain four values")
        self.weights = tuple(float(v) for v in weights)
        self.base = WeightedIoUBCELoss()

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Compute weighted deep-supervision loss.

        Inputs: outputs containing S3, S4, S5, Sg; target [B,1,H,W]. Output:
        scalar loss.
        """
        preds = [outputs["S3"], outputs["S4"], outputs["S5"], outputs["Sg"]]
        return sum(w * self.base(p, target) for w, p in zip(self.weights, preds))


@register_model("lgps_re")
class LGPsRE(nn.Module):
    """Tiny PraNet-inspired polyp segmentation model with pretrained features.

    Input: RGB [B,3,H,W]. Default output: logits [B,1,H,W]. The network combines
    a truncated ImageNet-pretrained MobileNetV3-Small encoder, parallel partial
    decoding, ECA-enhanced depthwise-separable reverse attention, and a low-level
    boundary branch. The architecture is designed to keep the complete trainable
    parameter count below one million rather than relying on a trainable-only trick.
    """

    def __init__(self, classes: int = 1, pretrained: bool = True,
                 ppd_channels: int = 32, ra_hidden: int = 24) -> None:
        super().__init__()
        if classes != 1:
            raise ValueError("LGPsRE is configured for binary polyp segmentation and requires classes=1.")
        self.encoder = MobileNetV3SmallTruncated(pretrained=pretrained)
        self.ppd = LitePPD(width=ppd_channels)
        self.global_pred = nn.Conv2d(ppd_channels, 1, 3, padding=1)
        self.ra5 = ReverseAttention(96, ra_hidden)
        self.ra4 = ReverseAttention(40, ra_hidden)
        self.ra3 = ReverseAttention(24, ra_hidden)
        self.detail = LowLevelBoundaryFusion(16, ra_hidden)
        self.final_pred = nn.Conv2d(ra_hidden, 1, 1)
        self.loss_fn = WeightedIoUBCELoss()
        self.deep_loss_fn = TinyDeepSupervisionLoss()
        self._verify_parameter_budget()

    def _verify_parameter_budget(self) -> None:
        """Assert that the complete trainable parameter set is below one million.

        Input: none. Output: none. Raises RuntimeError if the hard model-size budget
        is violated, making parameter growth impossible to overlook during iteration.
        """
        count = self.parameter_count()
        if count >= _MAX_TRAINABLE_PARAMS:
            raise RuntimeError(
                f"LGPsRE exceeds the hard trainable-parameter budget: {count:,} >= {_MAX_TRAINABLE_PARAMS:,}"
            )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """Run tiny PraNet-style segmentation.

        Input: x [B,3,H,W]. Output: logits [B,1,H,W] when return_attn=False;
        otherwise a dictionary containing final prediction, Sg, S5, S4, S3 and
        reverse-attention maps R5, R4 and R3.
        """
        size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.encoder(x)
        global_feature = self.ppd(f3, f4, f5)
        sg = self.global_pred(global_feature)
        s5, a5, r5 = self.ra5(f5, sg)
        s4, a4, r4 = self.ra4(f4, s5)
        s3, a3, r3 = self.ra3(f3, s4)
        final_feature = self.detail(f2, r3)
        out = self.final_pred(final_feature)
        out = _resize(out, size)
        if not return_attn:
            return out
        return {
            "pred": out,
            "Sg": _resize(sg, size),
            "S5": _resize(s5, size),
            "S4": _resize(s4, size),
            "S3": _resize(s3, size),
            "R5": _resize(a5, size),
            "R4": _resize(a4, size),
            "R3": _resize(a3, size),
        }

    def forward_with_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return prediction and intermediate reverse-attention maps.

        Input: x [B,3,H,W]. Output dictionary contains pred, Sg, R3, R4 and R5
        plus deep-supervision logits S3, S4 and S5 for visualization and Grad-CAM.
        """
        return self.forward(x, return_attn=True)

    def compute_loss(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Compute the four-stage deep-supervision loss.

        Inputs: outputs from forward(..., return_attn=True), target [B,1,H,W].
        Output: scalar weighted BCE plus weighted IoU loss.
        """
        return self.deep_loss_fn(outputs, target)

    def parameter_count(self) -> int:
        """Count all trainable parameters.

        Input: none. Output: integer equal to sum(p.numel() for trainable p).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_report(self) -> Dict[str, float]:
        """Return a compact parameter-budget report.

        Input: none. Output: dictionary with exact count, count in millions and
        a boolean indicating whether the hard one-million budget is satisfied.
        """
        count = self.parameter_count()
        return {
            "params": float(count),
            "params_m": float(count / 1e6),
            "under_1m": bool(count < _MAX_TRAINABLE_PARAMS),
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model.

    Input: model. Output: integer parameter count using requires_grad=True only.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_parameter_budget(model: nn.Module, limit: int = _MAX_TRAINABLE_PARAMS) -> int:
    """Validate the hard trainable-parameter budget.

    Input: model and parameter limit. Output: exact count. Raises AssertionError
    whenever the count is not strictly below the requested limit.
    """
    count = count_parameters(model)
    assert count < limit, f"Parameter budget exceeded: {count:,} >= {limit:,}"
    return count


def smoke_test(device: Optional[str] = None, batch_size: int = 4,
               image_size: int = 352, pretrained: bool = False) -> Tuple[int, Tuple[int, ...]]:
    """Run a budget and shape test for the tiny model.

    Input: device, batch size, image size and pretrained flag. Output: parameter
    count and prediction shape. The default uses random weights so it is offline-safe.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LGPsRE(pretrained=pretrained).to(device).eval()
    params = verify_parameter_budget(model)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (batch_size, 1, image_size, image_size), tuple(y.shape)
    return params, tuple(y.shape)


if __name__ == "__main__":
    model = LGPsRE(pretrained=False)
    params = count_parameters(model)
    print(f"Trainable Params: {params:,}")
    print(f"Trainable Params (M): {params / 1e6:.6f}")
    print(f"Under 1M: {params < 1_000_000}")
    shape = smoke_test(pretrained=False)
    print(f"Smoke test: params={shape[0]:,}, output={shape[1]}")


__all__ = [
    "LGPsRE",
    "MobileNetV3SmallTruncated",
    "LitePPD",
    "ReverseAttention",
    "WeightedIoUBCELoss",
    "TinyDeepSupervisionLoss",
    "count_parameters",
    "verify_parameter_budget",
    "smoke_test",
]
