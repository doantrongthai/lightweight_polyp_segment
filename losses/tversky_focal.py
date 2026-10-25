"""
losses/tversky_focal.py
────────────────────────────────────────────────────────────────
Tversky Loss + Focal Loss for ablation against the original Weighted BCE +
Weighted IoU (structure loss).

Why this combination beats structure loss on polyp data
(EEPES 2025 ablation on Kvasir-SEG, 12 losses compared):
  • Tversky (α=0.3, β=0.7): penalises false negatives more than
    false positives — crucial for small/flat polyps where missing
    a lesion is far worse than over-segmenting background.
  • Focal (γ=2.0): down-weights easy background pixels so the
    gradient signal comes from hard boundary / small-polyp regions.
  • Combined Dice on Kvasir-SEG: 0.8917  vs  structure loss ~0.883.

Hyperparameter choices
──────────────────────
  α=0.3, β=0.7   — from the paper's experimental optimum;
                    β > α means FN is penalised more than FP.
  γ=2.0           — standard focal exponent (Lin et al., 2017).
  focal_alpha=0.25 — foreground weighting; 0.25 works well when
                     background >> foreground (class imbalance).
  smooth=1e-8     — numerical stability only (not a tunable param).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss


def tversky_loss(pred: torch.Tensor, mask: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-8) -> torch.Tensor:
    """
    Tversky loss for binary segmentation.

    TI  = (TP + s) / (TP + α·FP + β·FN + s)
    TL  = 1 - mean(TI over batch)

    α + β = 1 is conventional but not required.
    Setting β > α penalises missed detections more than
    false alarms — the right trade-off for polyp screening.
    """
    p = torch.sigmoid(pred)                              # [B,1,H,W]
    tp = (p * mask).sum(dim=(1, 2, 3))
    fp = (p * (1.0 - mask)).sum(dim=(1, 2, 3))
    fn = ((1.0 - p) * mask).sum(dim=(1, 2, 3))
    ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - ti).mean()


def focal_loss(pred: torch.Tensor, mask: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """
    Sigmoid focal loss (Lin et al., RetinaNet 2017).

    FL(p) = -α (1-p_t)^γ log(p_t)

    Reduces the relative loss for well-classified easy pixels and
    focuses training on hard, misclassified examples — exactly the
    challenging boundaries and small-polyp centres.
    """
    bce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    p_t  = torch.exp(-bce)                              # probability of correct class
    fl   = alpha * (1.0 - p_t) ** gamma * bce
    return fl.mean()


def tversky_focal_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Combined loss = Tversky(α=0.3, β=0.7) + Focal(γ=2, α=0.25).

    Both terms are O(1) in magnitude so no manual weighting is needed.
    Binary segmentation ablation loss.
    """
    return tversky_loss(pred, mask) + focal_loss(pred, mask)


@register_loss("tversky_focal")
class TverskyFocalLoss(nn.Module):
    """Tversky + Focal loss for ablation."""

    def forward(self, pred, mask):
        return tversky_focal_loss(pred, mask)


@register_loss("tversky")
class TverskyLoss(nn.Module):
    """Tversky-only ablation."""

    def forward(self, pred, mask):
        return tversky_loss(pred, mask)


@register_loss("focal")
class FocalLoss(nn.Module):
    """Focal-only ablation."""

    def forward(self, pred, mask):
        return focal_loss(pred, mask)
