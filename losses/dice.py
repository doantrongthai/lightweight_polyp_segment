"""
losses/dice.py
────────────────────────────────────────────────────────────────
Standard soft Dice loss (Milletari et al., V-Net, 2016; Sudre et al.,
2017), used as the base coefficient for several other losses in this
package (arccos_dice.py, ac_region in dacmc.py, etc.).

    DSC  = (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)
    L    = 1 - DSC

where P is the soft prediction (sigmoid of the logits) and G is the
binary ground truth. `smooth` is a Laplace smoothing constant that
avoids a 0/0 division when both the prediction and the ground truth
are empty (e.g. a background-only training crop), and slightly damps
the gradient when the foreground is extremely small.

This module also defines the classic combined loss

    L = lambda_bce * BCE + lambda_dice * (1 - DSC)

which mixes the pixel-wise BCE term with the region-overlap-driven
Dice term — the most common baseline combo loss in medical image
segmentation, and the one several of the papers implemented in this
package (e.g. TK-BUB, DACMC) compare against directly.
"""

import torch
import torch.nn as nn

from . import register_loss

_SMOOTH = 1.0


def dice_coefficient(pred: torch.Tensor, mask: torch.Tensor,
                     smooth: float = _SMOOTH) -> torch.Tensor:
    """
    Soft Dice similarity coefficient per image in the batch.

    pred : raw logits, [B,C,H,W]
    mask : binary ground truth, [B,C,H,W]
    """
    prob = torch.sigmoid(pred)
    intersection = (prob * mask).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(pred: torch.Tensor, mask: torch.Tensor,
             smooth: float = _SMOOTH) -> torch.Tensor:
    """L = 1 - DSC, averaged over the batch."""
    dsc = dice_coefficient(pred, mask, smooth)
    return (1.0 - dsc).mean()


@register_loss("dice")
class DiceLoss(nn.Module):
    """
    Standard soft Dice loss.

    smooth : Laplace smoothing constant added to numerator/denominator
             of the Dice ratio (avoids 0/0 on empty prediction+mask).
    """

    def __init__(self, smooth: float = _SMOOTH):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        return dice_loss(pred, mask, self.smooth)


def dice_bce_loss(pred: torch.Tensor, mask: torch.Tensor,
                  lambda_bce: float = 1.0, lambda_dice: float = 1.0,
                  smooth: float = _SMOOTH) -> torch.Tensor:
    """L = lambda_bce * BCE + lambda_dice * (1 - DSC)."""
    bce = nn.functional.binary_cross_entropy_with_logits(pred, mask)
    dc = dice_loss(pred, mask, smooth)
    return lambda_bce * bce + lambda_dice * dc


@register_loss("dice_bce")
@register_loss("bce_dice")
class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss.

    lambda_bce, lambda_dice : weights for the pixel-wise BCE term and
                               the region-wise Dice term.
    smooth                  : passed through to the Dice ratio.
    """

    def __init__(self, lambda_bce: float = 1.0, lambda_dice: float = 1.0,
                 smooth: float = _SMOOTH):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.smooth = smooth

    def forward(self, pred, mask):
        return dice_bce_loss(pred, mask, self.lambda_bce,
                             self.lambda_dice, self.smooth)