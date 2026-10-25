
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from . import register_model


_MAX_TRAINABLE_PARAMS = 1_000_000


def _resize(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize a feature tensor.

    Input: [B,C,h,w]. Output: [B,C,H,W]. Bilinear interpolation is used to match
    the spatial alignment convention of the repository UNet implementation.
    """
    target = tuple(size)
    if x.shape[-2:] == target:
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=True)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize a tensor to the spatial size of a reference tensor.

    Input: x [B,C,h,w], ref [B,D,H,W]. Output: x resized to [B,C,H,W].
    """
    return _resize(x, ref.shape[-2:])


class ConvBNAct(nn.Module):
    """Compact convolutional projection.

    Input: [B,Cin,H,W]. Output: [B,Cout,H/s,W/s]. Batch normalization and SiLU
    provide compact nonlinear projection for encoder-decoder feature alignment.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
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
        """Apply projection.

        Input: [B,Cin,H,W]. Output: [B,Cout,H/s,W/s].
        """
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution.

    Input: [B,Cin,H,W]. Output: [B,Cout,H,W]. The depthwise spatial operator
    reduces computation while the pointwise operator preserves channel mixing.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: bool = True) -> None:
        super().__init__()
        self.dw = ConvBNAct(in_channels, in_channels, 3, groups=in_channels)
        self.pw = ConvBNAct(in_channels, out_channels, 1, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise and pointwise convolution.

        Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
        """
        return self.pw(self.dw(x))


class ECA(nn.Module):
    """Efficient Channel Attention.

    Input: [B,C,H,W]. Output: [B,C,H,W]. The parameter-efficient one-dimensional
    channel interaction preserves useful channel cues without an MLP bottleneck.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweight channels.

        Input: [B,C,H,W]. Output: [B,C,H,W].
        """
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class SpatialAttention(nn.Module):
    """Lightweight spatial attention.

    Input: [B,C,H,W]. Output: [B,1,H,W]. Channel-average and channel-maximum
    descriptors provide a low-parameter spatial gate for region and boundary cues.
    """

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate spatial gate.

        Input: [B,C,H,W]. Output: [B,1,H,W] in [0,1].
        """
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class BoundaryPriority(nn.Module):
    """Boundary-priority gate.

    Input: [B,C,H,W]. Output: [B,1,H,W]. A parameter-free Laplacian prior is
    combined with learned spatial attention to strengthen uncertain polyp borders.
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
        """Compute boundary-priority map.

        Input: [B,C,H,W]. Output: [B,1,H,W] with a bounded positive gate.
        """
        base = x.mean(dim=1, keepdim=True)
        edge = torch.abs(F.conv2d(base, self.laplacian, padding=1))
        edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        learned = self.spatial(x)
        return 0.75 + 0.35 * edge + 0.15 * learned


class LiteRFB(nn.Module):
    """Compact receptive-field fusion.

    Input: [B,Cin,H,W]. Output: [B,Cout,H,W]. Parallel depthwise branches with
    different dilation scales aggregate local detail and broader context.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch = max(out_channels // 4, 8)
        self.b1 = DepthwiseSeparableConv(in_channels, branch)
        self.b2 = nn.Sequential(
            ConvBNAct(in_channels, branch, 1),
            ConvBNAct(branch, branch, 3, groups=branch),
        )
        self.b3 = nn.Sequential(
            ConvBNAct(in_channels, branch, 1),
            nn.Conv2d(
                branch,
                branch,
                3,
                padding=5,
                dilation=5,
                groups=branch,
                bias=False,
            ),
            nn.BatchNorm2d(branch),
            nn.SiLU(inplace=True),
        )
        self.fuse = ConvBNAct(branch * 3, out_channels, 1)
        self.eca = ECA(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse multiscale receptive fields.

        Input: [B,Cin,H,W]. Output: [B,Cout,H,W].
        """
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.eca(self.fuse(y))


class LitePPD(nn.Module):
    """Higher-capacity lightweight Parallel Partial Decoder.

    Inputs: f3 [B,24,H3,W3], f4 [B,40,H4,W4], f5 [B,96,H5,W5]. Output:
    global feature [B,96,H3,W3]. The larger fusion width is reserved for global
    localization because the PPD global map controls every later reverse-attention
    stage.
    """

    def __init__(self, width: int = 96) -> None:
        super().__init__()
        self.p3 = ConvBNAct(24, width, 1)
        self.p4 = ConvBNAct(40, width, 1)
        self.p5 = ConvBNAct(96, width, 1)
        self.mix = DepthwiseSeparableConv(width * 3, width)
        self.rfb = LiteRFB(width, width)
        self.eca = ECA(width)

    def forward(
        self,
        f3: torch.Tensor,
        f4: torch.Tensor,
        f5: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse the three deep feature levels.

        Inputs: f3 [B,24,H3,W3], f4 [B,40,H4,W4], f5 [B,96,H5,W5].
        Output: [B,96,H3,W3].
        """
        p3 = self.p3(f3)
        p4 = _resize_like(self.p4(f4), f3)
        p5 = _resize_like(self.p5(f5), f3)
        y = self.mix(torch.cat([p3, p4, p5], dim=1))
        return self.eca(self.rfb(y) + y)


class ReverseAttention(nn.Module):
    """Capacity-balanced PraNet reverse-attention stage.

    Input: feature Fi [B,C,H,W] and coarse logits S [B,1,h,w]. Outputs are
    refined logits Si [B,1,H,W], attention Ai [B,1,H,W], and refined feature
    Ri [B,D,H,W]. The core reverse complement remains Ai=1-sigmoid(S), while
    boundary and spatial gates provide targeted refinement.
    """

    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.pre = DepthwiseSeparableConv(in_channels, hidden)
        self.eca = ECA(hidden)
        self.spatial = SpatialAttention(5)
        self.boundary = BoundaryPriority()
        self.refine = DepthwiseSeparableConv(hidden, hidden)
        self.pred = nn.Conv2d(hidden, 1, 3, padding=1, bias=True)

    def forward(
        self,
        feature: torch.Tensor,
        previous_prediction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine the complementary region.

        Inputs: feature [B,C,H,W], previous_prediction [B,1,h,w]. Outputs are
        prediction [B,1,H,W], attention [B,1,H,W], and feature [B,D,H,W].
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


class DetailFusion(nn.Module):
    """Final multiscale detail fusion.

    Inputs: f1 [B,16,H1,W1], f2 [B,16,H2,W2], r3 [B,48,H3,W3]. Output:
    [B,48,H3,W3]. Fine encoder features are injected only at the final stage to
    preserve boundary detail without creating a large decoder pyramid.
    """

    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.f1 = DepthwiseSeparableConv(16, 8)
        self.f2 = DepthwiseSeparableConv(16, 16)
        self.fuse = ConvBNAct(8 + 16 + hidden, hidden, 1)
        self.eca = ECA(hidden)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        r3: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse shallow and reverse-attention detail.

        Inputs: f1 [B,16,H1,W1], f2 [B,16,H2,W2], r3 [B,48,H3,W3].
        Output: [B,48,H3,W3].
        """
        p1 = _resize_like(self.f1(f1), r3)
        p2 = _resize_like(self.f2(f2), r3)
        return self.eca(self.fuse(torch.cat([p1, p2, r3], dim=1)) + r3)


class MobileNetV3SmallPretrained(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small encoder trunk.

    Input: RGB [B,3,H,W]. Outputs f1..f5 with channels (16,16,24,40,96) at
    approximate strides (2,4,8,16,32). The extra semantic block at the deepest
    level is retained while the classifier and final 576-channel expansion are
    removed to keep the full model under one million trainable parameters.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.blocks = nn.Sequential(*list(base.features[:12]))
        self.out_channels = (16, 16, 24, 40, 96)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Extract hierarchical feature maps.

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
        f5 = self.blocks[11](f5)
        return f1, f2, f3, f4, f5


class WeightedIoUBCELoss(nn.Module):
    """PraNet-style weighted BCE plus weighted IoU with boundary emphasis.

    Input: logits [B,1,H,W], target [B,1,H,W]. Output: scalar loss. The
    local-average discrepancy supplies structural weighting while a Laplacian
    factor gives additional emphasis to difficult mask transitions.
    """

    def __init__(
        self,
        gamma: float = 5.0,
        pool_kernel: int = 31,
        boundary_strength: float = 1.0,
    ) -> None:
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
        pad = self.pool_kernel // 2
        local = F.avg_pool2d(target, self.pool_kernel, stride=1, padding=pad)
        weight = 1.0 + self.gamma * torch.abs(local - target)
        edge = torch.abs(F.conv2d(target, self.laplacian, padding=1))
        edge = edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        weight = weight * (1.0 + self.boundary_strength * edge)
        wbce = (
            weight * F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            )
        ).sum(dim=(1, 2, 3)) / (weight.sum(dim=(1, 2, 3)) + 1e-6)
        inter = (weight * pred * target).sum(dim=(1, 2, 3))
        union = (
            weight * (pred + target - pred * target)
        ).sum(dim=(1, 2, 3))
        wiou = 1.0 - (inter + 1.0) / (union + 1.0)
        return (wbce + wiou).mean()


class DeepSupervisionLoss(nn.Module):
    """Weighted four-stage supervision for the PraNet-style prediction hierarchy.

    Inputs: dictionary containing S3, S4, S5 and Sg plus target [B,1,H,W]. Output:
    scalar loss. The stage weights favor the final high-resolution map while keeping
    gradients active in global and reverse-attention maps.
    """

    def __init__(self, weights: Sequence[float] = (0.45, 0.25, 0.2, 0.1)) -> None:
        super().__init__()
        if len(weights) != 4:
            raise ValueError("weights must contain four values")
        self.weights = tuple(float(v) for v in weights)
        self.base = WeightedIoUBCELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep-supervision loss.

        Inputs: outputs with S3, S4, S5, Sg and target [B,1,H,W]. Output: scalar.
        """
        preds = [outputs["S3"], outputs["S4"], outputs["S5"], outputs["Sg"]]
        return sum(
            weight * self.base(prediction, target)
            for weight, prediction in zip(self.weights, preds)
        )


@register_model("lgps_optimized")
class LGPsOptimized(nn.Module):
    """Optimized sub-million-parameter PraNet-inspired polyp segmentation model.

    Input: RGB [B,3,H,W]. Default output: logits [B,1,H,W]. The network uses a
    pretrained MobileNetV3-Small semantic trunk, an expanded but depthwise-separable
    PPD, capacity-balanced reverse-attention stages, fine-detail fusion, and a
    deep-fusion output so that the repository's single-output training interface
    still backpropagates through multiple prediction levels.
    """

    def __init__(
        self,
        classes: int = 1,
        pretrained: bool = True,
        ppd_channels: int = 96,
        ra5_hidden: int = 64,
        ra4_hidden: int = 56,
        ra3_hidden: int = 48,
    ) -> None:
        super().__init__()
        if classes != 1:
            raise ValueError(
                "LGPsRE is configured for binary polyp segmentation and requires classes=1."
            )
        self.encoder = MobileNetV3SmallPretrained(pretrained=pretrained)
        self.ppd = LitePPD(width=ppd_channels)
        self.global_pred = nn.Conv2d(ppd_channels, 1, 3, padding=1)
        self.ra5 = ReverseAttention(96, ra5_hidden)
        self.ra4 = ReverseAttention(40, ra4_hidden)
        self.ra3 = ReverseAttention(24, ra3_hidden)
        self.detail = DetailFusion(hidden=ra3_hidden)
        self.final_pred = nn.Conv2d(ra3_hidden, 1, 1)
        self.loss_fn = WeightedIoUBCELoss()
        self.deep_loss_fn = DeepSupervisionLoss()
        self.fusion_logits = nn.Parameter(
            torch.tensor([2.0, 1.0, 0.0, -1.0, -1.0], dtype=torch.float32)
        )
        self._verify_parameter_budget()

    def _verify_parameter_budget(self) -> None:
        """Verify the strict trainable-parameter budget.

        Input: none. Output: none. Raises RuntimeError when the complete trainable
        parameter count is not strictly below one million.
        """
        count = self.parameter_count()
        if count >= _MAX_TRAINABLE_PARAMS:
            raise RuntimeError(
                f"LGPsRE exceeds the hard trainable-parameter budget: "
                f"{count:,} >= {_MAX_TRAINABLE_PARAMS:,}"
            )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        """Run the optimized segmentation network.

        Input: x [B,3,H,W]. Output: logits [B,1,H,W] for normal training. When
        return_attn=True, a dictionary includes prediction, Sg, S5, S4, S3,
        R5, R4 and R3 at input resolution.
        """
        size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.encoder(x)
        global_feature = self.ppd(f3, f4, f5)
        sg = self.global_pred(global_feature)
        s5, a5, r5 = self.ra5(f5, sg)
        s4, a4, r4 = self.ra4(f4, s5)
        s3, a3, r3 = self.ra3(f3, s4)
        final_feature = self.detail(f1, f2, r3)
        detail_logits = self.final_pred(final_feature)

        outputs = [
            _resize(detail_logits, size),
            _resize(s3, size),
            _resize(s4, size),
            _resize(s5, size),
            _resize(sg, size),
        ]
        weights = torch.softmax(self.fusion_logits, dim=0)
        out = sum(weight * prediction for weight, prediction in zip(weights, outputs))

        if not return_attn:
            return out

        return {
            "pred": out,
            "Sg": outputs[4],
            "S5": outputs[3],
            "S4": outputs[2],
            "S3": outputs[1],
            "R5": _resize(a5, size),
            "R4": _resize(a4, size),
            "R3": _resize(a3, size),
        }

    def forward_with_attention(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return segmentation and intermediate visualization tensors.

        Input: x [B,3,H,W]. Output: dictionary with pred, Sg, S3, S4, S5, R3,
        R4 and R5. The normal forward path does not retain these tensors unless
        the caller explicitly requests attention output.
        """
        return self.forward(x, return_attn=True)

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep-supervision weighted BCE plus weighted IoU.

        Inputs: outputs dictionary from forward_with_attention and target
        [B,1,H,W]. Output: scalar training loss.
        """
        return self.deep_loss_fn(outputs, target)

    def parameter_count(self) -> int:
        """Count all trainable parameters.

        Input: none. Output: integer equal to sum(p.numel() for p in model
        parameters when requires_grad=True).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_report(self) -> Dict[str, float]:
        """Return the strict parameter-budget report.

        Input: none. Output: exact parameter count, count in millions, and a
        boolean showing whether the model is strictly below one million.
        """
        count = self.parameter_count()
        return {
            "params": float(count),
            "params_m": float(count / 1e6),
            "under_1m": bool(count < _MAX_TRAINABLE_PARAMS),
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters.

    Input: PyTorch model. Output: integer number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_parameter_budget(
    model: nn.Module,
    limit: int = _MAX_TRAINABLE_PARAMS,
) -> int:
    """Assert that the trainable parameter count is below a strict limit.

    Input: model and integer limit. Output: exact parameter count. An AssertionError
    is raised whenever the count is not strictly below the limit.
    """
    count = count_parameters(model)
    assert count < limit, f"Parameter budget exceeded: {count:,} >= {limit:,}"
    return count


def smoke_test(
    device: Optional[str] = None,
    batch_size: int = 4,
    image_size: int = 352,
    pretrained: bool = False,
) -> Tuple[int, Tuple[int, ...]]:
    """Run parameter and shape validation.

    Input: device, batch size, image size and pretrained flag. Output: parameter
    count and prediction shape. Random weights are used by default for offline tests.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LGPsRE(pretrained=pretrained).to(device).eval()
    params = verify_parameter_budget(model)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (batch_size, 1, image_size, image_size), tuple(y.shape)
    return params, tuple(y.shape)


def parameter_report(model: nn.Module) -> Dict[str, float]:
    """Return exact parameter statistics.

    Input: model. Output: parameter count in absolute and million units plus
    the hard-budget status.
    """
    count = count_parameters(model)
    return {
        "params": float(count),
        "params_m": float(count / 1e6),
        "under_1m": float(count < _MAX_TRAINABLE_PARAMS),
    }


if __name__ == "__main__":
    model = LGPsRE(pretrained=False)
    params = count_parameters(model)
    print(f"Trainable Params: {params:,}")
    print(f"Trainable Params (M): {params / 1e6:.6f}")
    print(f"Under 1M: {params < 1_000_000}")
    shape = smoke_test(pretrained=False)
    print(f"Smoke test: params={shape[0]:,}, output={shape[1]}")


__all__ = [
    "LGPsOptimized",
    "MobileNetV3SmallPretrained",
    "LitePPD",
    "ReverseAttention",
    "WeightedIoUBCELoss",
    "DeepSupervisionLoss",
    "count_parameters",
    "verify_parameter_budget",
    "parameter_report",
    "smoke_test",
]
