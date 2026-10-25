"""
losses/nac.py
────────────────────────────────────────────────────────────────
Normalized Active Contour (NAC) loss
(Ho, Nguyen, Tran, Pham — "LiteMamba-Bound: A lightweight Mamba-based
model with boundary-aware and normalized active contour loss for skin
lesion segmentation", Methods 235 (2025) 10-25).

NAC extends the Active Contour loss (Chen et al., 2019) by normalising
its Chan-Vese-style region term with a denominator inspired by the IoU
loss (Eq. 19):

    NAC(y, yhat) = [ sum yhat*(c1-y)^2 + sum (1-yhat)*(c2-y)^2 ]
                   -----------------------------------------------
                          sum ( y^2 + yhat^2 - y*yhat )

Unlike the classic IoU-loss denominator (y + yhat), the paper uses
(y^2 + yhat^2) in place of (y + yhat) — since both live in [0, 1],
squaring sharpens the contrast between confidently-foreground and
confidently-background pixels, which the authors report increases the
loss's sensitivity to error regions near the boundary.

As in the paper's own Active-Contour-based losses (and DACMC, another
paper implemented in this package), c1/c2 are fixed foreground/
background constants for the supervised setting: c1 = 1, c2 = 0
(Section 3.6).

A boundary-smoothness length term (Eq. 20, an L1 total-variation-style
penalty on the prediction) is added on top:

    Lengthterm = (1 / (W*H)) * sum( |yhat[i+1,j]-yhat[i,j]|
                                   + |yhat[i,j+1]-yhat[i,j]| )

giving the final combined loss (Eq. 21):

    L_total = NAC + lambda * Lengthterm,   lambda = 1e-4 (paper's value)

Numerical stability
─────────────────────
The NAC denominator sum(y^2 + yhat^2 - y*yhat) is a sum of per-pixel
terms that can be rewritten as (y - yhat)^2 + y*yhat, which is always
>= 0 and is exactly 0 only when a pixel has y = yhat = 0. Consequently
the whole-image denominator only reaches exactly 0 in the fully
degenerate case of an all-background ground truth with a perfect
all-zero prediction — rare, but a real edge case (e.g. an empty
background-only training crop late in training). We therefore add a
modest epsilon to the denominator, the same defensive pattern used for
every other ratio-based loss in this package.
"""

import torch
import torch.nn as nn

from . import register_loss

_EPS = 1e-4
_LAMBDA = 1e-4   # paper's reported value for the length-term weight


def nac_loss(pred: torch.Tensor, mask: torch.Tensor,
            c1: float = 1.0, c2: float = 0.0, eps: float = _EPS) -> torch.Tensor:
    """Normalized Active Contour region term (Eq. 19), averaged over the batch."""
    prob = torch.sigmoid(pred)

    numerator = (prob * (c1 - mask) ** 2 + (1.0 - prob) * (c2 - mask) ** 2)
    numerator = numerator.sum(dim=(1, 2, 3))

    denominator = (mask ** 2 + prob ** 2 - mask * prob).sum(dim=(1, 2, 3))

    return (numerator / (denominator + eps)).mean()


def length_term(pred: torch.Tensor) -> torch.Tensor:
    """Boundary-smoothness length term (Eq. 20), averaged over the batch."""
    prob = torch.sigmoid(pred)
    h, w = prob.shape[-2], prob.shape[-1]

    diff_h = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs().sum(dim=(1, 2, 3))
    diff_w = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs().sum(dim=(1, 2, 3))

    return ((diff_h + diff_w) / (h * w)).mean()


def nac_total_loss(pred: torch.Tensor, mask: torch.Tensor,
                   lambda_: float = _LAMBDA, c1: float = 1.0, c2: float = 0.0,
                   eps: float = _EPS) -> torch.Tensor:
    """Combined loss (Eq. 21): NAC region term + lambda * length term."""
    return nac_loss(pred, mask, c1, c2, eps) + lambda_ * length_term(pred)


@register_loss("nac")
class NACLoss(nn.Module):
    """
    Normalized Active Contour loss (Ho et al., 2025).

    lambda_ : weight of the boundary length term (Eq. 21). Default 1e-4,
              the value used throughout the paper's experiments.
    c1, c2  : fixed foreground/background constants used by the region
              term in the supervised setting (Section 3.6).
    """

    def __init__(self, lambda_: float = _LAMBDA, c1: float = 1.0,
                 c2: float = 0.0, eps: float = _EPS):
        super().__init__()
        self.lambda_ = lambda_
        self.c1 = c1
        self.c2 = c2
        self.eps = eps

    def forward(self, pred, mask):
        return nac_total_loss(pred, mask, self.lambda_, self.c1, self.c2, self.eps)


@register_loss("nac_region")
class NACRegionLoss(nn.Module):
    """NAC region term only (Eq. 19), without the length term — ablation."""

    def __init__(self, c1: float = 1.0, c2: float = 0.0, eps: float = _EPS):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.eps = eps

    def forward(self, pred, mask):
        return nac_loss(pred, mask, self.c1, self.c2, self.eps)
