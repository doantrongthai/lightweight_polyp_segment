"""
losses/tk_bub.py
────────────────────────────────────────────────────────────────
Tversky-Kahneman Baroni-Urbani-Buser (TK-BUB) loss
(Nham, Trinh, Nguyen, Pham, Tran — "An EfficientNet-encoder U-Net Joint
Residual Refinement Module with Tversky-Kahneman Baroni-Urbani-Buser
loss for biomedical image Segmentation", Biomedical Signal Processing
and Control 83 (2023) 104631).

Construction (paper Section 3.2):

  1. Baroni-Urbani-Buser (BUB) coefficient, written in terms of the
     (soft) confusion-matrix rates (Eq. 6):

         BUB = (TP + sqrt(TP*TN)) / (TP + FP + FN + sqrt(TP*TN))

     Unlike Dice/Jaccard, BUB explicitly folds the true-negative rate
     TN into its numerator (via sqrt(TP*TN)), which is what the
     authors exploit to counter class imbalance.

  2. Two BUB-based losses (Eqs. 7-8):

         l1 = 1 - BUB = (FP+FN) / (TP+FP+FN + sqrt(TP*TN))
         l2 =           (FP+FN) / (TN+FP+FN + sqrt(TP*TN))

     l2 swaps the TP term in the denominator for TN, further
     down-weighting the (typically huge) true-negative contribution
     relative to TP — the paper shows this gives faster convergence.

  3. Both l1, l2 are passed through the Tversky-Kahneman probability
     weighting function (Eq. 9, from prospect theory):

         omega(x) = x^g / [x^g + (1-x)^g]^(1/g)

     giving the final TK-BUB losses (Eqs. 10-11):

         L_TK-BUB-v1 = omega(l1),   L_TK-BUB-v2 = omega(l2)

     with g = gamma = 4/3 reported as the best-performing exponent
     (Section 3.2.2). The paper's own ablation (Table 7) shows
     L_TK-BUB-v2 consistently outperforming l1, l2 and v1 alone, so it
     is used as the default/primary loss here, matching the authors'
     choice for their end-to-end training.

Numerical stability
─────────────────────
  • TP, FP, FN, TN are computed as *pixel-averaged rates* (mean rather
    than raw pixel counts) so that all four quantities live in a
    predictable, bounded [0, 1] range regardless of image resolution
    or batch size — this makes a single epsilon value meaningful
    everywhere, rather than needing to be re-scaled per image size.
    Since l1/l2 are ratios of sums that are all scaled by the same
    constant (1/N), using rates instead of raw counts does not change
    their value, only the numerical scale we're regularising.
  • sqrt(TP*TN) is the one place with a genuine instability: d/dx
    sqrt(x) diverges as x -> 0, and TP -> 0 is a routine situation
    early in training on a class-imbalanced mask (few foreground
    pixels correctly predicted yet). We therefore evaluate
    sqrt(TP*TN + eps) instead of the literal sqrt(TP*TN), with a
    modest eps (not an infinitesimal one) — the same fix applied to
    every division/sqrt/exp-based instability found earlier in this
    project (FRS loss, DACMC curvature term, arccos-Dice).
  • gamma = 4/3 > 1 is not just an empirical choice in the paper — it
    also happens to keep the Tversky-Kahneman weighting function's own
    gradient well-behaved at l=0 and l=1 (d/dl l^gamma = gamma*l^(gamma-1)
    -> 0 as l -> 0 when gamma>1, rather than exploding), so no
    additional clamping is required there. We still clamp l1/l2 into
    (0, 1) with a small margin purely as a guard against floating-point
    overshoot at the exact boundaries.
"""

import torch
import torch.nn as nn

from . import register_loss

_EPS = 1e-4          # floor inside sqrt(TP*TN) and ratio denominators
_L_MARGIN = 1e-6     # keeps l1/l2 strictly inside (0, 1) against fp overshoot
_GAMMA = 4.0 / 3.0   # Tversky-Kahneman exponent (paper's reported optimum)


def confusion_rates(pred: torch.Tensor, mask: torch.Tensor):
    """
    Soft, pixel-averaged confusion-matrix rates (TP, FP, FN, TN) per
    image in the batch.

    pred : raw logits, [B,C,H,W]
    mask : binary ground truth, [B,C,H,W]
    """
    prob = torch.sigmoid(pred)
    tp = (prob * mask).mean(dim=(1, 2, 3))
    fp = (prob * (1.0 - mask)).mean(dim=(1, 2, 3))
    fn = ((1.0 - prob) * mask).mean(dim=(1, 2, 3))
    tn = ((1.0 - prob) * (1.0 - mask)).mean(dim=(1, 2, 3))
    return tp, fp, fn, tn


def bub_l1(pred: torch.Tensor, mask: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """BUB loss version 1 = 1 - BUB (Eq. 7)."""
    tp, fp, fn, tn = confusion_rates(pred, mask)
    root = torch.sqrt(tp * tn + eps)
    return (fp + fn) / (tp + fp + fn + root + eps)


def bub_l2(pred: torch.Tensor, mask: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """BUB loss version 2, with TN replacing TP in the denominator (Eq. 8)."""
    tp, fp, fn, tn = confusion_rates(pred, mask)
    root = torch.sqrt(tp * tn + eps)
    return (fp + fn) / (tn + fp + fn + root + eps)


def tversky_kahneman_weight(l: torch.Tensor, gamma: float = _GAMMA,
                            eps: float = _EPS) -> torch.Tensor:
    """Tversky-Kahneman probability-weighting function omega(x) (Eq. 9)."""
    l = torch.clamp(l, min=_L_MARGIN, max=1.0 - _L_MARGIN)
    num = l ** gamma
    denom = (l ** gamma + (1.0 - l) ** gamma + eps) ** (1.0 / gamma)
    return num / (denom + eps)


def tk_bub_v1_loss(pred: torch.Tensor, mask: torch.Tensor,
                   gamma: float = _GAMMA, eps: float = _EPS) -> torch.Tensor:
    """L_TK-BUB-v1 = omega(l1) (Eq. 10), averaged over the batch."""
    l1 = bub_l1(pred, mask, eps)
    return tversky_kahneman_weight(l1, gamma, eps).mean()


def tk_bub_v2_loss(pred: torch.Tensor, mask: torch.Tensor,
                   gamma: float = _GAMMA, eps: float = _EPS) -> torch.Tensor:
    """L_TK-BUB-v2 = omega(l2) (Eq. 11), averaged over the batch — the
    version the authors adopt for their end-to-end training."""
    l2 = bub_l2(pred, mask, eps)
    return tversky_kahneman_weight(l2, gamma, eps).mean()


@register_loss("tk_bub")
@register_loss("tk_bub_v2")
class TKBUBLossV2(nn.Module):
    """
    TK-BUB loss, version 2 (Eq. 11) — the paper's recommended default.

    gamma : Tversky-Kahneman weighting exponent. Default 4/3, the value
            the paper reports as the best-performing choice tested over
            gamma in (1, 2) (Section 3.2.2).
    """

    def __init__(self, gamma: float = _GAMMA, eps: float = _EPS):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, mask):
        return tk_bub_v2_loss(pred, mask, self.gamma, self.eps)


@register_loss("tk_bub_v1")
class TKBUBLossV1(nn.Module):
    """TK-BUB loss, version 1 (Eq. 10)."""

    def __init__(self, gamma: float = _GAMMA, eps: float = _EPS):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, mask):
        return tk_bub_v1_loss(pred, mask, self.gamma, self.eps)


@register_loss("bub_v1")
class BUBLossV1(nn.Module):
    """Plain BUB loss, l1 = 1 - BUB (Eq. 7), without the TK weighting — ablation."""

    def __init__(self, eps: float = _EPS):
        super().__init__()
        self.eps = eps

    def forward(self, pred, mask):
        return bub_l1(pred, mask, self.eps).mean()


@register_loss("bub_v2")
class BUBLossV2(nn.Module):
    """Plain BUB loss, l2 (Eq. 8), without the TK weighting — ablation."""

    def __init__(self, eps: float = _EPS):
        super().__init__()
        self.eps = eps

    def forward(self, pred, mask):
        return bub_l2(pred, mask, self.eps).mean()
