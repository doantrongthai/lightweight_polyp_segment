import math
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError as exc:
    raise ImportError("torchvision is required for LGPs with EfficientNet-B0.") from exc

from . import register_model


def _resize(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Resize a feature map to (H, W) with bilinear interpolation.

    Input: x [B, C, h, w]. Output: [B, C, H, W]. The operation follows the
    interpolation convention used by the supplied UNet skeleton.
    """
    target = tuple(size)
    if x.shape[-2:] == target:
        return x
    return F.interpolate(x, size=target, mode="bilinear", align_corners=True)


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize x to the spatial size of ref.

    Input: x [B, C, h, w], ref [B, D, H, W]. Output: [B, C, H, W].
    """
    return _resize(x, ref.shape[-2:])


class ConvBNAct(nn.Module):
    """Compact convolution block.

    Input: [B, Cin, H, W]. Output: [B, Cout, H/s, W/s]. It provides the
    projection and nonlinear transformation used throughout LGPs.
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
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization and optional activation.

        Input: [B, Cin, H, W]. Output: [B, Cout, H/s, W/s].
        """
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution for low-cost decoder computation.

    Input: [B, Cin, H, W]. Output: [B, Cout, H, W]. The design replaces a
    dense 3x3 convolution with depthwise spatial filtering plus pointwise
    channel mixing, following lightweight CNN practice.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw = ConvBNAct(in_channels, in_channels, 3, groups=in_channels)
        self.pw = ConvBNAct(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise and pointwise convolutions.

        Input: [B, Cin, H, W]. Output: [B, Cout, H, W].
        """
        return self.pw(self.dw(x))


class SpatialAttention(nn.Module):
    """Lightweight spatial attention based on pooled channel descriptors.

    Input: [B, C, H, W]. Output: [B, 1, H, W]. The module follows the
    spatial-attention idea of CBAM while keeping a single 7x7 convolution.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a spatial attention map.

        Input: [B, C, H, W]. Output: [B, 1, H, W] with values in [0, 1].
        """
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class MultiScaleRFB(nn.Module):
    """Lightweight receptive-field block using parallel dilated branches.

    Input: [B, Cin, H, W]. Output: [B, Cout, H, W]. It approximates the
    receptive-field aggregation principle of receptive-field blocks while
    reducing decoder cost through depthwise-separable convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = max(out_channels // 4, 8)
        self.b1 = DepthwiseSeparableConv(in_channels, branch_channels)
        self.b2 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, 1),
            ConvBNAct(branch_channels, branch_channels, 3, groups=branch_channels),
        )
        self.b3 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, 1),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                3,
                padding=3,
                dilation=3,
                groups=branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, 1),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                3,
                padding=5,
                dilation=5,
                groups=branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = ConvBNAct(branch_channels * 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate multi-scale receptive fields.

        Input: [B, Cin, H, W]. Output: [B, Cout, H, W].
        """
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))


class FeatureCalibration(nn.Module):
    """Residual channel-gating block for decoder feature calibration.

    Input: [B, C, H, W]. Output: [B, C, H, W]. It performs lightweight
    residual recalibration inspired by channel-refinement mechanisms.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.reduce = ConvBNAct(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrate feature responses using global channel statistics.

        Input: [B, C, H, W]. Output: [B, C, H, W].
        """
        pooled = F.adaptive_avg_pool2d(x, 1)
        gate = torch.sigmoid(self.expand(self.reduce(pooled)))
        return x * gate + x


class BoundaryPriorityModule(nn.Module):
    """Boundary-priority estimator used to gate reverse attention.

    Input: [B, C, H, W]. Output: [B, 1, H, W]. The module predicts a soft
    boundary-priority map that is multiplied with the PraNet reverse
    attention weight.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            DepthwiseSeparableConv(in_channels, max(in_channels // 2, 16)),
            SpatialAttention(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a soft boundary-priority map.

        Input: [B, C, H, W]. Output: [B, 1, H, W].
        """
        return self.body(x)


class LaplacianBoundaryWeight(nn.Module):
    """Differentiable Laplacian boundary estimator for loss weighting.

    Input: [B, 1, H, W]. Output: [B, 1, H, W]. The fixed Laplacian response
    emphasizes local mask transitions and provides an additional boundary
    weighting signal for supervised segmentation loss.
    """

    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """Compute a normalized boundary-aware weight map.

        Input: [B, 1, H, W]. Output: [B, 1, H, W] with baseline weight 1.
        """
        boundary = torch.abs(F.conv2d(target, self.kernel, padding=1))
        boundary = boundary / (boundary.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return 1.0 + boundary


class HighPrecisionPartialFusion(nn.Module):
    """High-precision partial feature fusion for the global decoder map.

    Input: three tensors [B, Ci, Hi, Wi] at progressively coarser resolutions.
    Output: [B, C, H3, W3]. It implements a compact parallel fusion inspired
    by PraNet's PPD while using depthwise-separable convolutions.
    """

    def __init__(self, channels: Sequence[int], out_channels: int = 64) -> None:
        super().__init__()
        self.proj = nn.ModuleList(
            [ConvBNAct(ch, out_channels, 1) for ch in channels]
        )
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(out_channels * 3, out_channels),
            MultiScaleRFB(out_channels, out_channels),
            FeatureCalibration(out_channels),
        )

    def forward(
        self,
        f3: torch.Tensor,
        f4: torch.Tensor,
        f5: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse f3, f4 and f5 in parallel.

        Inputs: f3 [B,C3,H3,W3], f4 [B,C4,H4,W4], f5 [B,C5,H5,W5].
        Output: [B,C,H3,W3].
        """
        p3 = self.proj[0](f3)
        p4 = _resize_like(self.proj[1](f4), f3)
        p5 = _resize_like(self.proj[2](f5), f3)
        return self.fuse(torch.cat([p3, p4, p5], dim=1))


class ReverseAttentionBlock(nn.Module):
    """PraNet-style reverse attention block with compact spatial attention.

    Input: feature F_i [B,C,H,W] and next prediction S_{i+1} [B,1,h,w].
    Output: refined prediction S_i [B,1,H,W], reverse weight Ai [B,1,H,W],
    and refined feature R_i [B,C,H,W]. The reverse weight follows
    Ai = 1 - sigmoid(upsample(S_{i+1})) and is multiplied by BPM output.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.pre = ConvBNAct(in_channels, hidden_channels, 3)
        self.spatial = SpatialAttention()
        self.pred = nn.Sequential(
            DepthwiseSeparableConv(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        self.bpm = BoundaryPriorityModule(hidden_channels)

    def forward(
        self,
        feature: torch.Tensor,
        previous_prediction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine a coarse prediction using reverse and boundary attention.

        Inputs: feature [B,C,H,W], previous_prediction [B,1,h,w].
        Outputs: prediction [B,1,H,W], attention [B,1,H,W], feature [B,D,H,W].
        """
        x = self.pre(feature)
        x = x * self.spatial(x)
        previous = _resize_like(previous_prediction, x)
        ai = 1.0 - torch.sigmoid(previous)
        bpm = self.bpm(x)
        reverse_weight = ai * bpm
        refined = x * reverse_weight
        prediction = self.pred(refined)
        return prediction, reverse_weight, refined


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder exposing five segmentation feature levels.

    Input: RGB tensor [B,3,352,352]. Outputs f1..f5 with progressively
    coarser spatial resolutions. EfficientNet-B0 ImageNet initialization is
    used when pretrained=True.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.out_channels = (24, 40, 80, 112, 192)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract five hierarchical feature maps.

        Input: [B,3,H,W]. Outputs f1..f5 approximately at 1/8, 1/16, 1/32,
        1/32 and 1/64 input resolution for EfficientNet-B0.
        """
        f1 = self.features[2](self.features[1](self.features[0](x)))
        f2 = self.features[3](f1)
        f3 = self.features[4](f2)
        f4 = self.features[5](f3)
        f5 = self.features[6](f4)
        return f1, f2, f3, f4, f5


class WeightedIoUBCELoss(nn.Module):
    """PraNet-style weighted IoU plus weighted BCE segmentation loss.

    Inputs: prediction logits [B,1,H,W], target [B,1,H,W]. Output: scalar
    loss. The spatial weight uses local average pooling and can optionally
    include a Laplacian boundary factor.
    """

    def __init__(
        self,
        gamma: float = 5.0,
        kernel_size: int = 31,
        boundary_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.kernel_size = kernel_size
        self.boundary_scale = boundary_scale
        self.boundary = LaplacianBoundaryWeight()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted BCE and weighted IoU.

        Inputs: logits [B,1,H,W], target [B,1,H,W]. Output: scalar tensor.
        """
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = _resize_like(logits, target)
        prediction = torch.sigmoid(logits)
        pad = self.kernel_size // 2
        local = F.avg_pool2d(target, self.kernel_size, stride=1, padding=pad)
        weight = 1.0 + self.gamma * torch.abs(local - target)
        if self.boundary_scale > 0:
            boundary_weight = self.boundary(target)
            weight = weight * (1.0 + self.boundary_scale * (boundary_weight - 1.0))
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        wbce = (weight * bce).sum(dim=(1, 2, 3)) / (weight.sum(dim=(1, 2, 3)) + 1e-6)
        intersection = (weight * prediction * target).sum(dim=(1, 2, 3))
        union = (weight * (prediction + target - prediction * target)).sum(dim=(1, 2, 3))
        wiou = 1.0 - (intersection + 1.0) / (union + 1.0)
        return (wbce + wiou).mean()


@register_model("lgps")
class LGPs(nn.Module):
    """Lightweight PraNet-inspired polyp segmentation network.

    Input: RGB image [B,3,H,W]. Default output: [B,1,H,W]. The architecture
    combines EfficientNet-B0, compact PPD, spatially enhanced reverse
    attention, boundary-priority gating, and deep supervision.
    """

    def __init__(
        self,
        classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        if classes != 1:
            raise ValueError("LGPs is configured for binary polyp segmentation and requires classes=1.")
        self.encoder = EfficientNetEncoder(pretrained=pretrained)
        self.ppd = HighPrecisionPartialFusion(self.encoder.out_channels[2:], decoder_channels)
        self.global_pred = nn.Conv2d(decoder_channels, classes, 3, padding=1)
        self.ra5 = ReverseAttentionBlock(self.encoder.out_channels[4], decoder_channels // 2)
        self.ra4 = ReverseAttentionBlock(self.encoder.out_channels[3], decoder_channels // 2)
        self.ra3 = ReverseAttentionBlock(self.encoder.out_channels[2], decoder_channels // 2)
        self.loss_fn = WeightedIoUBCELoss()

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Run LGPs segmentation.

        Input: x [B,3,H,W]. Output: [B,1,H,W] when return_attn=False.
        When return_attn=True, a dictionary contains Sg, S5, S4, S3 and
        R5, R4, R3 attention maps.
        """
        size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.encoder(x)
        global_feature = self.ppd(f3, f4, f5)
        sg = self.global_pred(global_feature)
        s5, r5, a5 = self.ra5(f5, sg)
        s4, r4, a4 = self.ra4(f4, s5)
        s3, r3, a3 = self.ra3(f3, s4)
        output = _resize(s3, size)
        if not return_attn:
            return output
        return {
            "Sg": _resize(sg, size),
            "S5": _resize(s5, size),
            "S4": _resize(s4, size),
            "S3": output,
            "R5": _resize(a5, size),
            "R4": _resize(a4, size),
            "R3": _resize(a3, size),
        }

    def forward_with_attention(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the network and return segmentation plus intermediate attention.

        Input: x [B,3,H,W]. Output dictionary contains prediction, Sg and
        reverse-attention maps R3, R4 and R5. This method uses the same forward
        graph and is intended for visualization or Grad-CAM extraction.
        """
        return self.forward(x, return_attn=True)

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        weights: Optional[Sequence[float]] = None,
    ) -> torch.Tensor:
        """Compute deep-supervision loss over Sg, S5, S4 and S3.

        Inputs: outputs from forward(..., return_attn=True), target [B,1,H,W].
        Output: scalar loss. Default supervision weights emphasize the final
        high-resolution prediction while retaining gradients at all stages.
        """
        if weights is None:
            weights = (0.4, 0.3, 0.2, 0.1)
        predictions = [outputs["S3"], outputs["S4"], outputs["S5"], outputs["Sg"]]
        if len(weights) != len(predictions):
            raise ValueError("weights must contain four values for S3, S4, S5 and Sg.")
        total = predictions[0].new_zeros(())
        for prediction, weight in zip(predictions, weights):
            total = total + float(weight) * self.loss_fn(prediction, target)
        return total

    def parameter_count(self) -> int:
        """Return the number of trainable parameters.

        Input: none. Output: integer parameter count.
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def overlay_segmentation(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.45,
    show: bool = True,
) -> plt.Figure:
    """Visualize an image with ground-truth and prediction contours.

    Inputs: image [H,W,3], ground_truth [H,W], prediction [H,W] or logits
    [H,W]. Output: Matplotlib Figure containing the original image,
    ground-truth contour, prediction contour and blended prediction mask.
    """
    if prediction.max() > 1.0 or prediction.min() < 0.0:
        prediction = 1.0 / (1.0 + np.exp(-prediction))
    gt = ground_truth > 0.5
    pred = prediction > threshold
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.imshow(pred.astype(np.float32), alpha=alpha)
    ax.contour(gt.astype(np.float32), levels=[0.5], linewidths=1.5)
    ax.contour(pred.astype(np.float32), levels=[0.5], linewidths=1.5, linestyles="--")
    ax.axis("off")
    ax.set_title("Ground Truth vs Prediction")
    if show:
        plt.show()
    return fig


def visualize_attention(
    image: np.ndarray,
    attention: Dict[str, np.ndarray | torch.Tensor],
    show: bool = True,
) -> plt.Figure:
    """Display global and reverse-attention maps beside the input image.

    Inputs: image [H,W,3] and attention maps containing Sg, R3, R4 and R5.
    Output: Matplotlib Figure with one image panel and four attention panels.
    """
    keys = ["Sg", "R3", "R4", "R5"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("Input")
    axes[0].axis("off")
    for axis, key in zip(axes[1:], keys):
        value = attention[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().float().numpy()
        value = np.squeeze(value)
        axis.imshow(value)
        axis.set_title(key)
        axis.axis("off")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters.

    Input: PyTorch model. Output: integer count of trainable parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def benchmark_forward(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 4,
    image_size: int = 352,
) -> Tuple[Tuple[int, ...], float]:
    """Run a shape and latency smoke test for LGPs.

    Inputs: model, device name, batch size and square image size. Output:
    prediction shape tuple and average forward time in milliseconds.
    """
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
        end = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
        if start is not None and end is not None:
            start.record()
            y = model(x)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            import time
            begin = time.perf_counter()
            y = model(x)
            elapsed = (time.perf_counter() - begin) * 1000.0
    return tuple(y.shape), float(elapsed)


__all__ = [
    "LGPs",
    "WeightedIoUBCELoss",
    "EfficientNetEncoder",
    "HighPrecisionPartialFusion",
    "ReverseAttentionBlock",
    "BoundaryPriorityModule",
    "LaplacianBoundaryWeight",
    "overlay_segmentation",
    "visualize_attention",
    "count_parameters",
    "benchmark_forward",
]

