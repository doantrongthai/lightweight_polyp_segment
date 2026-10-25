"""
losses/dacmc.py
────────────────────────────────────────────────────────────────
Deep Active Contour and Mean Curvature (DACMC) loss
(Zhai, Pang, Zheng, Li, Hou — "Medical Image Segmentation based on Deep
Active Contour and Mean Curvature Loss Function", arXiv:2607.12586).

DACMC combines a Chan-Vese-style region (active-contour) term with a
mean-curvature regularisation term that is approximated cheaply with a
fixed 3x3 convolution kernel, avoiding the need to back-propagate
through explicit high-order derivatives:

    L_DACMC = sum_{i,j} [ u_ij (c1 - v_ij)^2 + (1-u_ij)(c2 - v_ij)^2 ]
              + lambda * C                                        (Eq. 16)

    C = sum_{i,j} | u*_ij * v_ij / (v*_ij + eps) |                (Eq. 17)

where u is the network's soft prediction, v is the (binary) ground
truth, and u*, v* are curvature maps obtained by convolving u and v
with the fixed kernel of Eq. (19):

    H(x) ≈ Kernel * x,   Kernel = [[-1/16,  5/16, -1/16],
                                    [ 5/16, -1   ,  5/16],
                                    [-1/16,  5/16, -1/16]]

Implementation notes
─────────────────────
  • Region term (c1, c2). In the general (unsupervised, CV-style)
    formulation c1/c2 are running means updated every iteration
    (Eq. 20). The paper explicitly states that "In the supervised
    learning framework, the method also fixes the energy of c1 and c2
    as foreground and background, simply defined as constants" — the
    same convention used earlier in the paper's ACE loss (Section
    II.C), where c1=1 (foreground) and c2=0 (background). We follow
    that convention here.
  • Curvature term ratio (Eq. 17 / Fig. 1: "prediction_cur * label /
    (label_cur + eps)"). v* (the ground-truth curvature map) is exactly
    zero over any flat interior/exterior region of the mask and only
    departs from zero near the mask boundary, so v* legitimately
    crosses through zero. Dividing by (v* + eps) with an infinitesimal
    eps — as in the literal Eq. (17) — lets the denominator pass
    arbitrarily close to zero from either side, which blows up the
    gradient w.r.t. u* by a factor of 1/eps right at those crossings.
    We saw exactly this failure mode (NaN losses) once already with a
    different fuzzy-set loss in this package, so here we use
    |v*| clamped to a modest floor as the denominator instead of the
    literal v*+eps, and additionally clamp the resulting ratio's
    magnitude before averaging. Both changes only affect pixels where
    the literal formula would already be numerically degenerate, and
    keep the loss finite and its gradient bounded everywhere.
  • The curvature kernel is a fixed (non-trainable) buffer, so it moves
    with the module across devices/dtypes automatically.
  • lambda (the region/curvature trade-off) is not given a single
    numeric value in the paper ("we get a suitable interval after
    experiments"); we default to lambda=1.0 and expose it as a
    constructor argument for tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss

_EPS = 1e-2          # denominator floor for the curvature ratio (Eq. 17)
_RATIO_CLAMP = 50.0  # bound on |curvature ratio| before averaging

_KERNEL = torch.tensor([
    [-1 / 16,  5 / 16, -1 / 16],
    [ 5 / 16, -1.0,     5 / 16],
    [-1 / 16,  5 / 16, -1 / 16],
], dtype=torch.float32)


def _mean_curvature(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Fast mean-curvature approximation H(x) ~= Kernel * x (Eqs. 18-19),
    applied independently per channel via a depthwise convolution.

    x      : [B, C, H, W]
    kernel : [3, 3] curvature-approximation kernel
    """
    c = x.shape[1]
    weight = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1).to(dtype=x.dtype, device=x.device)
    return F.conv2d(x, weight, padding=1, groups=c)


def ac_region_loss(pred: torch.Tensor, mask: torch.Tensor,
                   c1: float = 1.0, c2: float = 0.0) -> torch.Tensor:
    """
    Chan-Vese-style active-contour region term (first summand of Eq. 16),
    with c1/c2 fixed to the supervised foreground/background constants.

    pred : raw logits, [B,C,H,W]
    mask : binary ground truth, [B,C,H,W]
    """
    u = torch.sigmoid(pred)
    region = u * (c1 - mask) ** 2 + (1.0 - u) * (c2 - mask) ** 2
    return region.mean(dim=(1, 2, 3)).mean()


def curvature_loss(pred: torch.Tensor, mask: torch.Tensor,
                   kernel: torch.Tensor = _KERNEL) -> torch.Tensor:
    """
    Mean-curvature regularisation term C (Eq. 17), computed with the
    fast convolution-kernel approximation of Eqs. (18)-(19).
    """
    u = torch.sigmoid(pred)
    u_star = _mean_curvature(u, kernel)
    v_star = _mean_curvature(mask, kernel)

    denom = v_star.abs().clamp_min(_EPS)
    ratio = (u_star * mask) / denom
    ratio = torch.clamp(ratio.abs(), max=_RATIO_CLAMP)

    return ratio.mean(dim=(1, 2, 3)).mean()


def dacmc_loss(pred: torch.Tensor, mask: torch.Tensor,
              lambda_: float = 1.0, c1: float = 1.0, c2: float = 0.0,
              kernel: torch.Tensor = _KERNEL) -> torch.Tensor:
    """Combined DACMC loss (Eq. 16): region term + lambda * curvature term."""
    region = ac_region_loss(pred, mask, c1, c2)
    curv = curvature_loss(pred, mask, kernel)
    return region + lambda_ * curv


@register_loss("dacmc")
class DACMCLoss(nn.Module):
    """
    Deep Active Contour and Mean Curvature loss (Zhai et al., 2026).

    lambda_ : trade-off weight between the region term and the
              curvature term (Eq. 16). Default 1.0 — tune per dataset
              (the paper reports a "suitable interval" rather than one
              fixed value; too large or too small degrades results).
    c1, c2  : fixed foreground/background constants used by the region
              term in the supervised setting (Section IV.A).
    """

    def __init__(self, lambda_: float = 1.0, c1: float = 1.0, c2: float = 0.0):
        super().__init__()
        self.lambda_ = lambda_
        self.c1 = c1
        self.c2 = c2
        self.register_buffer("kernel", _KERNEL.clone())

    def forward(self, pred, mask):
        return dacmc_loss(pred, mask, self.lambda_, self.c1, self.c2, self.kernel)


@register_loss("ac_region")
@register_loss("ac")
class ACRegionLoss(nn.Module):
    """Region-only term — the 'AC' baseline compared against in Table 1."""

    def __init__(self, c1: float = 1.0, c2: float = 0.0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def forward(self, pred, mask):
        return ac_region_loss(pred, mask, self.c1, self.c2)


@register_loss("mean_curvature")
class MeanCurvatureLoss(nn.Module):
    """Curvature-only term (Eq. 17) — useful for ablation studies."""

    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", _KERNEL.clone())

    def forward(self, pred, mask):
        return curvature_loss(pred, mask, self.kernel)
