"""
losses/arccos_dice_bce.py
────────────────────────────────────────────────────────────────
Combined Arccosine-Dice + BCE loss:

    L = lambda_bce * BCE + lambda_dice * (2/pi) * arccos(DSC)

A pixel-wise term (BCE) is combined with the region-overlap-driven
Arccosine Dice term (see losses/arccos_dice.py) — the same rationale as
the well-known "Dice + BCE" combo loss, just swapping the classic
`1 - DSC` Dice term for its arccos-transformed counterpart.

Both sub-losses are reused directly from their own modules (bce.py /
arccos_dice.py) rather than re-implemented here, so any numerical-
stability fix made in one place (e.g. the DSC clamping in
arccos_dice.py) automatically applies here too.
"""

import torch
import torch.nn as nn

from . import register_loss
from .arccos_dice import arccos_dice_loss, _SMOOTH, _EPS


def arccos_dice_bce_loss(pred: torch.Tensor, mask: torch.Tensor,
                         lambda_bce: float = 1.0, lambda_dice: float = 1.0,
                         smooth: float = _SMOOTH, eps: float = _EPS) -> torch.Tensor:
    """L = lambda_bce * BCE + lambda_dice * (2/pi) * arccos(DSC)."""
    bce = nn.functional.binary_cross_entropy_with_logits(pred, mask)
    ac_dice = arccos_dice_loss(pred, mask, smooth, eps)
    return lambda_bce * bce + lambda_dice * ac_dice


@register_loss("arccos_dice_bce")
@register_loss("bce_arccos_dice")
class ArccosDiceBCELoss(nn.Module):
    """
    Combined Arccosine-Dice + BCE loss.

    lambda_bce, lambda_dice : weights for the pixel-wise BCE term and
                               the region-wise arccos-Dice term.
    smooth, eps             : passed through to the Dice ratio / arccos
                               clamp (see arccos_dice.py).
    """

    def __init__(self, lambda_bce: float = 1.0, lambda_dice: float = 1.0,
                 smooth: float = _SMOOTH, eps: float = _EPS):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, mask):
        return arccos_dice_bce_loss(pred, mask, self.lambda_bce,
                                    self.lambda_dice, self.smooth, self.eps)
