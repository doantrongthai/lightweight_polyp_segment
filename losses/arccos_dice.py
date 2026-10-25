"""
losses/arccos_dice.py
────────────────────────────────────────────────────────────────
Arccosine Dice loss (v2 — boundary-aware, gradient-safe, annealable):

    L = (2 / pi) * arccos(DSC^gamma)

where DSC is the (soft) Dice similarity coefficient between the
prediction and the ground truth. Since arccos(1) = 0 and
arccos(0) = pi/2, this maps DSC in [0, 1] onto a loss in [0, 1]
that is 0 for a perfect match and 1 for no overlap at all — same
range and monotonicity as the classic `1 - DSC` Dice loss, but with a
different (steeper near DSC=1, flatter near DSC=0) gradient shape.

────────────────────────────────────────────────────────────────
WHY v2 EXISTS — lessons from v1
────────────────────────────────────────────────────────────────
v1 of this loss trained well but generalised worse than
structure_loss (structure.py) on held-out polyp test sets. Two root
causes were identified, and both are fixed below:

1. GRADIENT BLOW-UP NEAR DSC = 1 IS NOT THE SAME AS VALUE BLOW-UP.
   Clamping DSC into [eps, 1-eps] keeps the *loss value* finite, but
   the *gradient* magnitude, -(2/pi)*gamma*DSC^(gamma-1) /
   sqrt(1 - DSC^(2*gamma)), still grows roughly like 1/sqrt(eps) right
   at the clamp boundary. Since DSC legitimately approaches 1 exactly
   when training is going best, this means the pixels/images the
   model has already learned well keep receiving *larger* gradient
   than plain Dice's `1 - DSC` would give them (which shrinks to 0).
   Plain Dice's vanishing gradient near DSC=1 is itself a mild,
   implicit regulariser — arccos was fighting that regulariser and
   pushing the model to over-fit training-set specifics (annotation
   style, imaging artifacts) right when it should be backing off.
   Fix: `SafeArccos`, a custom autograd Function that clips the local
   gradient magnitude directly (a soft, Huber-style saturation),
   instead of only clamping the input value.

2. THE ARCCOS-DICE TERM WAS BLIND TO THE BOUNDARY, UNLIKE
   structure_loss. structure_loss (PraNet/Polyp-PVT) uses the *same*
   boundary weight map for both its BCE term and its IoU term, so
   the region-overlap term itself is pulled toward getting boundary
   pixels right — exactly where polyp segmentation is hardest and
   matters most. v1's `arccos_dice_wbce_loss` only applied boundary
   weighting to the BCE half and left the Dice/arccos half as a
   plain, global, boundary-blind ratio, so the model could inflate
   global DSC via easy interior pixels without being pushed on the
   boundary through that term.
   Fix: `dice_coefficient(..., weight=...)` now accepts the same
   boundary weight map used elsewhere in this codebase, and
   `arccos_dice_loss` computes it by default (`boundary_weighted=True`).

Everything else (Laplace smoothing, the eps clamp, the gamma
temperature knob, the combined BCE/boundary losses) is retained from
v1 and still works the same way; see the docstrings below.

Numerical stability
─────────────────────
DSC is still clamped strictly inside (-1, 1) — with a modest margin,
not an infinitesimal one — before calling arccos. That clamp is now a
secondary safety net; the primary defense against exploding gradients
is the gradient clip in `SafeArccos` (see above).

Temperature (gamma)
─────────────────────
arccos(DSC) has a fixed, non-tunable gradient shape: very steep near
DSC=1, comparatively flat near DSC=0. There is no knob to make the
loss react more or less aggressively to a still-bad prediction. We add
a temperature exponent gamma:

    L = (2 / pi) * arccos(DSC^gamma)

  - gamma = 1 recovers the original loss exactly.
  - gamma > 1 pushes DSC^gamma closer to 0 whenever DSC < 1, which
    flattens the loss (and its gradient) in the low/mid-DSC region —
    useful if training is unstable early on, or (new in v2) as a way
    to blunt the over-fitting behaviour described above without
    reaching for annealing or gradient clipping.
  - gamma < 1 pushes DSC^gamma closer to 1 for any DSC > 0, which
    steepens the loss earlier. Since DSC^gamma for gamma < 1 has a
    derivative that itself diverges as DSC -> 0, the eps clamp is
    two-sided ([eps, 1-eps]) so the loss stays well-behaved for any
    gamma > 0, not only gamma >= 1.

Annealing (new in v2)
─────────────────────
`arccos_dice_loss_annealed` linearly blends arccos-Dice (early
training, when its steeper-near-1 gradient helps push DSC up fast)
towards plain `1 - DSC` (late training, when its vanishing gradient
near DSC=1 acts as a brake against over-fitting):

    alpha = 1 - epoch / total_epochs        (1 -> 0 over training)
    L = alpha * arccos_dice + (1 - alpha) * plain_dice

This requires the caller to pass the current epoch, so it is exposed
as a plain function rather than an nn.Module with fixed state.

Multi-scale DSC (new in v2)
─────────────────────
`multiscale_dice_coefficient` averages the (optionally boundary-
weighted) DSC computed at the native resolution and at one or more
downsampled resolutions. This makes the loss less sensitive to exact
pixel-level boundary placement — useful across polyp datasets whose
annotators drew boundaries with different levels of care/precision —
without discarding boundary sensitivity entirely (native scale is
still included and still boundary-weighted).

Combined losses
─────────────────────
Deviating away from a plain BCE, this module also defines a boundary-
weighted BCE (the same weighting scheme PraNet/Polyp-PVT's structure
loss uses in structure.py: pixels near the mask boundary get up to 6x
the weight of pixels deep in a flat region, since boundaries are where
segmentation errors matter most and are hardest to get right), and
combines it with the Arccosine Dice term above:

    L = lambda_bce * WeightedBCE + lambda_dice * (2/pi) * arccos(DSC)

Since v2's arccos_dice_loss is boundary-weighted by default, this
combined loss now double-reinforces the boundary in both halves,
matching structure_loss's design more closely than v1 did.

It also defines a genuine boundary loss (Sobel edge matching): unlike
the weighted BCE above, which only reweights an existing pixel-wise
loss near the boundary, this term extracts an edge map from both the
prediction and the mask (via a Sobel filter) and penalises the
difference between the two edge maps directly — it cares whether the
predicted contour sits in the right place and has the right shape,
independent of interior region overlap:

    L = lambda_dice * arccos_dice_loss + lambda_boundary * boundary_loss

Numerical note: the edge magnitude sqrt(Gx^2 + Gy^2) has the same
derivative blow-up at 0 as every other unguarded sqrt() in this
project (any pixel deep inside a flat region has Gx = Gy = 0), so it
is evaluated as sqrt(Gx^2 + Gy^2 + eps) instead. Also, unlike
arccos_dice_loss, boundary_loss is NOT bounded to [0, 1] (Sobel
magnitudes are unnormalised), so lambda_boundary typically needs to be
set lower (e.g. 0.1-0.3) than lambda_dice to keep the two terms
balanced — check the relative magnitude on your own data before
picking a default.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss

_SMOOTH = 1.0     # Laplace smoothing constant for the Dice ratio
_EPS = 1e-4       # margin kept away from the +-1 poles of arccos
_WBCE_KERNEL = 31  # boundary-detection window for the weighted BCE / DSC term
_BOUNDARY_EPS = 1e-6  # floor inside the Sobel edge-magnitude sqrt
_MAX_GRAD = 10.0  # default gradient-magnitude clip for SafeArccos


# ────────────────────────────────────────────────────────────────
# Gradient-safe arccos
# ────────────────────────────────────────────────────────────────

class _SafeArccosFn(torch.autograd.Function):
    """
    arccos(x) with the *local gradient magnitude* clipped, rather than
    only clamping x away from +-1.

    d/dx arccos(x) = -1 / sqrt(1 - x^2), which diverges as x -> +-1.
    Clamping x keeps arccos(x) itself finite, but right at the clamp
    boundary the gradient is still huge — exactly the regime DSC sits
    in once training is going well. This Function instead saturates
    the gradient smoothly (Huber-style): full accuracy away from the
    poles, bounded magnitude near them.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, max_grad: float):
        ctx.save_for_backward(x)
        ctx.max_grad = max_grad
        return torch.arccos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        denom = torch.sqrt(torch.clamp(1.0 - x * x, min=1e-8))
        local_grad = torch.clamp(-1.0 / denom, min=-ctx.max_grad, max=ctx.max_grad)
        return grad_output * local_grad, None


def safe_arccos(x: torch.Tensor, max_grad: float = _MAX_GRAD) -> torch.Tensor:
    """arccos(x) with clipped backward gradient. See `_SafeArccosFn`."""
    return _SafeArccosFn.apply(x, max_grad)


# ────────────────────────────────────────────────────────────────
# Boundary weight map (shared with weighted BCE)
# ────────────────────────────────────────────────────────────────

def boundary_weight_map(mask: torch.Tensor, kernel_size: int = _WBCE_KERNEL) -> torch.Tensor:
    """
    Same weighting scheme as structure.py's PraNet/Polyp-PVT structure
    loss: pixels near the mask boundary (where the local average of
    the mask differs most from the pixel's own value) get up to 6x the
    weight of pixels deep inside a flat foreground/background region.
    """
    return 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=kernel_size, stride=1,
                     padding=kernel_size // 2) - mask
    )


# ────────────────────────────────────────────────────────────────
# Dice coefficient (now optionally boundary-weighted / multi-scale)
# ────────────────────────────────────────────────────────────────

def dice_coefficient(pred: torch.Tensor, mask: torch.Tensor,
                     smooth: float = _SMOOTH,
                     weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Soft Dice similarity coefficient per image in the batch.

    pred   : raw logits, [B,C,H,W]
    mask   : binary ground truth, [B,C,H,W]
    weight : optional per-pixel weight map, same shape as mask (e.g.
             from `boundary_weight_map`). When given, both the
             intersection and union are weighted, so boundary pixels
             contribute proportionally more to the ratio — matching
             how structure_loss's IoU term is weighted.
    """
    prob = torch.sigmoid(pred)
    if weight is None:
        intersection = (prob * mask).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    else:
        intersection = (prob * mask * weight).sum(dim=(1, 2, 3))
        union = ((prob + mask) * weight).sum(dim=(1, 2, 3))
    return (2.0 * intersection + smooth) / (union + smooth)


def boundary_weighted_dice_coefficient(pred: torch.Tensor, mask: torch.Tensor,
                                       smooth: float = _SMOOTH,
                                       kernel_size: int = _WBCE_KERNEL) -> torch.Tensor:
    """Convenience wrapper: DSC weighted by `boundary_weight_map(mask)`."""
    weight = boundary_weight_map(mask, kernel_size)
    return dice_coefficient(pred, mask, smooth, weight=weight)


def multiscale_dice_coefficient(pred: torch.Tensor, mask: torch.Tensor,
                                smooth: float = _SMOOTH,
                                scales: Sequence[float] = (1.0, 0.5, 0.25),
                                boundary_weighted: bool = True,
                                kernel_size: int = _WBCE_KERNEL) -> torch.Tensor:
    """
    Average DSC across `scales` (1.0 = native resolution). Downscaling
    is done with bilinear interpolation on both pred (post-sigmoid
    handled inside dice_coefficient via logits — so we interpolate the
    logits/mask directly) and mask. Reduces sensitivity to exact
    pixel-level boundary placement, which varies across polyp datasets
    with different annotation styles.
    """
    total = None
    for s in scales:
        if s == 1.0:
            p, m = pred, mask
        else:
            p = F.interpolate(pred, scale_factor=s, mode="bilinear", align_corners=False)
            m = F.interpolate(mask, scale_factor=s, mode="bilinear", align_corners=False)
        if boundary_weighted:
            w = boundary_weight_map(m, kernel_size)
            dsc = dice_coefficient(p, m, smooth, weight=w)
        else:
            dsc = dice_coefficient(p, m, smooth)
        total = dsc if total is None else total + dsc
    return total / len(scales)


# ────────────────────────────────────────────────────────────────
# Arccosine Dice loss
# ────────────────────────────────────────────────────────────────

def arccos_dice_loss(pred: torch.Tensor, mask: torch.Tensor,
                     smooth: float = _SMOOTH, eps: float = _EPS,
                     gamma: float = 1.0,
                     boundary_weighted: bool = True,
                     kernel_size: int = _WBCE_KERNEL,
                     max_grad: float = _MAX_GRAD,
                     multiscale: bool = False,
                     scales: Sequence[float] = (1.0, 0.5, 0.25)) -> torch.Tensor:
    """
    L = (2/pi) * safe_arccos(DSC^gamma), with DSC clamped to [eps, 1-eps]
    and the arccos gradient itself clipped (see `safe_arccos`).

    boundary_weighted : if True (default, new in v2), DSC is computed
                         with the same boundary weight map structure_loss
                         uses, so the region-overlap term is pulled
                         toward the boundary too, not just the BCE half
                         of any combined loss.
    max_grad           : gradient-magnitude clip passed to safe_arccos.
                          Lower = more conservative near DSC=1.
    multiscale          : if True, use `multiscale_dice_coefficient`
                          instead of a single-resolution DSC.
    gamma = 1.0 (default) reproduces the original, un-tempered loss
    shape (modulo the v2 gradient-safety and boundary-weighting fixes).
    """
    if multiscale:
        dsc = multiscale_dice_coefficient(pred, mask, smooth, scales,
                                          boundary_weighted, kernel_size)
    elif boundary_weighted:
        dsc = boundary_weighted_dice_coefficient(pred, mask, smooth, kernel_size)
    else:
        dsc = dice_coefficient(pred, mask, smooth)

    dsc = torch.clamp(dsc, min=eps, max=1.0 - eps)
    dsc_t = dsc if gamma == 1.0 else dsc ** gamma
    loss = (2.0 / math.pi) * safe_arccos(dsc_t, max_grad)
    return loss.mean()


def arccos_dice_loss_annealed(pred: torch.Tensor, mask: torch.Tensor,
                              epoch: int, total_epochs: int,
                              smooth: float = _SMOOTH, eps: float = _EPS,
                              boundary_weighted: bool = True,
                              kernel_size: int = _WBCE_KERNEL,
                              max_grad: float = _MAX_GRAD) -> torch.Tensor:
    """
    Linearly anneal from arccos-Dice (epoch 0, alpha=1) to plain
    `1 - DSC` (epoch >= total_epochs, alpha=0). Uses arccos's steeper
    gradient early on to push DSC up fast, then relies on plain Dice's
    vanishing gradient near DSC=1 as a brake against over-fitting late
    in training. `epoch` / `total_epochs` are 0-indexed and caller-
    supplied since this loss has no persistent state of its own.
    """
    if boundary_weighted:
        dsc = boundary_weighted_dice_coefficient(pred, mask, smooth, kernel_size)
    else:
        dsc = dice_coefficient(pred, mask, smooth)
    dsc = torch.clamp(dsc, min=eps, max=1.0 - eps)

    alpha = max(0.0, 1.0 - epoch / max(total_epochs, 1))
    arccos_term = (2.0 / math.pi) * safe_arccos(dsc, max_grad)
    plain_term = 1.0 - dsc
    return (alpha * arccos_term + (1.0 - alpha) * plain_term).mean()


@register_loss("arccos_dice")
@register_loss("dice_arccos")
class ArccosDiceLoss(nn.Module):
    """
    Arccosine Dice loss: L = (2/pi) * safe_arccos(DSC^gamma).

    smooth            : Laplace smoothing constant added to
                         numerator/denominator of the Dice ratio.
    eps               : margin kept between DSC and {0, 1} before
                         exponentiating and calling arccos.
    gamma             : temperature exponent applied to DSC before
                         arccos (see module docstring). Default 1.0.
    boundary_weighted : weight the DSC ratio itself by the same
                         boundary map structure_loss uses (default
                         True — this is the v2 default; set False to
                         recover v1's global, boundary-blind DSC).
    kernel_size       : window used to detect boundary pixels when
                         boundary_weighted=True.
    max_grad          : gradient-magnitude clip for the arccos
                         backward pass (see `safe_arccos`). Lower is
                         more conservative near DSC=1.
    multiscale        : average DSC over `scales` instead of a single
                         resolution (see `multiscale_dice_coefficient`).
    scales            : resolutions used when multiscale=True.
    """

    def __init__(self, smooth: float = _SMOOTH, eps: float = _EPS,
                 gamma: float = 1.0, boundary_weighted: bool = True,
                 kernel_size: int = _WBCE_KERNEL, max_grad: float = _MAX_GRAD,
                 multiscale: bool = False,
                 scales: Sequence[float] = (1.0, 0.5, 0.25)):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.gamma = gamma
        self.boundary_weighted = boundary_weighted
        self.kernel_size = kernel_size
        self.max_grad = max_grad
        self.multiscale = multiscale
        self.scales = scales

    def forward(self, pred, mask):
        return arccos_dice_loss(pred, mask, self.smooth, self.eps, self.gamma,
                                self.boundary_weighted, self.kernel_size,
                                self.max_grad, self.multiscale, self.scales)


# ────────────────────────────────────────────────────────────────
# Weighted BCE (unchanged from v1)
# ────────────────────────────────────────────────────────────────

def weighted_bce_loss(pred: torch.Tensor, mask: torch.Tensor,
                      kernel_size: int = _WBCE_KERNEL) -> torch.Tensor:
    """
    Boundary-weighted BCE: pixels near the mask boundary get up to 6x
    the weight of pixels deep inside a flat foreground or background
    region. Same weighting scheme as structure.py's PraNet/Polyp-PVT
    structure loss (now also shared by `boundary_weighted_dice_coefficient`
    via `boundary_weight_map`).
    """
    weight = boundary_weight_map(mask, kernel_size)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    return ((weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))).mean()


def arccos_dice_wbce_loss(pred: torch.Tensor, mask: torch.Tensor,
                          lambda_bce: float = 1.0, lambda_dice: float = 1.0,
                          smooth: float = _SMOOTH, eps: float = _EPS,
                          kernel_size: int = _WBCE_KERNEL,
                          boundary_weighted: bool = True,
                          max_grad: float = _MAX_GRAD) -> torch.Tensor:
    """
    L = lambda_bce * WeightedBCE + lambda_dice * (2/pi) * arccos(DSC).

    boundary_weighted defaults to True here (v2 change): both halves
    of this combined loss now see the boundary weight map, matching
    structure_loss's design instead of leaving the Dice half global.
    """
    wbce = weighted_bce_loss(pred, mask, kernel_size)
    ac_dice = arccos_dice_loss(pred, mask, smooth, eps,
                               boundary_weighted=boundary_weighted,
                               kernel_size=kernel_size, max_grad=max_grad)
    return lambda_bce * wbce + lambda_dice * ac_dice


@register_loss("arccos_dice_wbce")
@register_loss("wbce_arccos_dice")
class ArccosDiceWeightedBCELoss(nn.Module):
    """
    Combined Arccosine-Dice + boundary-weighted-BCE loss.

    lambda_bce, lambda_dice : weights for the weighted-BCE term and the
                               arccos-Dice term.
    kernel_size             : window used to detect boundary pixels for
                               the BCE weighting (see weighted_bce_loss).
    smooth, eps             : passed through to the Dice ratio / arccos
                               clamp (see arccos_dice_loss).
    boundary_weighted       : also weight the Dice/arccos half by the
                               boundary map (default True, v2 change).
    max_grad                : gradient-magnitude clip for the arccos
                               backward pass.
    """

    def __init__(self, lambda_bce: float = 1.0, lambda_dice: float = 1.0,
                 smooth: float = _SMOOTH, eps: float = _EPS,
                 kernel_size: int = _WBCE_KERNEL,
                 boundary_weighted: bool = True,
                 max_grad: float = _MAX_GRAD):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.smooth = smooth
        self.eps = eps
        self.kernel_size = kernel_size
        self.boundary_weighted = boundary_weighted
        self.max_grad = max_grad

    def forward(self, pred, mask):
        return arccos_dice_wbce_loss(pred, mask, self.lambda_bce,
                                     self.lambda_dice, self.smooth,
                                     self.eps, self.kernel_size,
                                     self.boundary_weighted, self.max_grad)


# ────────────────────────────────────────────────────────────────
# Boundary loss (Sobel edge matching) — unchanged from v1
# ────────────────────────────────────────────────────────────────

_SOBEL_KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_SOBEL_KY = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])


def _sobel_edge_magnitude(x: torch.Tensor, eps: float = _BOUNDARY_EPS) -> torch.Tensor:
    """Per-pixel Sobel gradient magnitude, applied depthwise per channel."""
    c = x.shape[1]
    kx = _SOBEL_KX.view(1, 1, 3, 3).repeat(c, 1, 1, 1).to(dtype=x.dtype, device=x.device)
    ky = _SOBEL_KY.view(1, 1, 3, 3).repeat(c, 1, 1, 1).to(dtype=x.dtype, device=x.device)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx ** 2 + gy ** 2 + eps)


def boundary_loss(pred: torch.Tensor, mask: torch.Tensor,
                  eps: float = _BOUNDARY_EPS) -> torch.Tensor:
    """Mean absolute difference between the Sobel edge maps of prob and mask."""
    prob = torch.sigmoid(pred)
    edge_prob = _sobel_edge_magnitude(prob, eps)
    edge_mask = _sobel_edge_magnitude(mask, eps)
    return (edge_prob - edge_mask).abs().mean(dim=(1, 2, 3)).mean()


def arccos_dice_boundary_loss(pred: torch.Tensor, mask: torch.Tensor,
                              lambda_dice: float = 1.0, lambda_boundary: float = 1.0,
                              smooth: float = _SMOOTH, eps: float = _EPS,
                              boundary_eps: float = _BOUNDARY_EPS,
                              boundary_weighted: bool = True,
                              max_grad: float = _MAX_GRAD) -> torch.Tensor:
    """L = lambda_dice * arccos_dice_loss + lambda_boundary * boundary_loss."""
    ac_dice = arccos_dice_loss(pred, mask, smooth, eps,
                               boundary_weighted=boundary_weighted,
                               max_grad=max_grad)
    b = boundary_loss(pred, mask, boundary_eps)
    return lambda_dice * ac_dice + lambda_boundary * b


@register_loss("arccos_dice_boundary")
@register_loss("boundary_arccos_dice")
class ArccosDiceBoundaryLoss(nn.Module):
    """
    Combined Arccosine-Dice + Sobel boundary loss.

    lambda_dice, lambda_boundary : weights for the region (arccos-Dice)
                                    term and the edge-matching term.
                                    boundary_loss is NOT bounded to
                                    [0, 1] like arccos_dice_loss is, so
                                    lambda_boundary usually needs to be
                                    set lower (e.g. 0.1-0.3) to keep the
                                    two terms balanced.
    smooth, eps                  : passed through to the Dice ratio /
                                    arccos clamp.
    boundary_eps                 : floor inside the Sobel edge-magnitude
                                    sqrt (see module notes above).
    boundary_weighted            : also weight the Dice/arccos half by
                                    the boundary map (default True).
    max_grad                     : gradient-magnitude clip for the
                                    arccos backward pass.
    """

    def __init__(self, lambda_dice: float = 1.0, lambda_boundary: float = 0.2,
                 smooth: float = _SMOOTH, eps: float = _EPS,
                 boundary_eps: float = _BOUNDARY_EPS,
                 boundary_weighted: bool = True,
                 max_grad: float = _MAX_GRAD):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary
        self.smooth = smooth
        self.eps = eps
        self.boundary_eps = boundary_eps
        self.boundary_weighted = boundary_weighted
        self.max_grad = max_grad

    def forward(self, pred, mask):
        return arccos_dice_boundary_loss(pred, mask, self.lambda_dice,
                                         self.lambda_boundary, self.smooth,
                                         self.eps, self.boundary_eps,
                                         self.boundary_weighted, self.max_grad)