import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Weighted BCE + weighted IoU loss used by PraNet/Polyp-PVT for binary
    polyp segmentation.
    """
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weight).sum(dim=(2, 3))
    union = ((prob + mask) * weight).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

    return (wbce + wiou).mean()


@register_loss("structure")
@register_loss("weighted_bce_iou")
class StructureLoss(nn.Module):
    """Weighted BCE + weighted IoU loss from PraNet/Polyp-PVT."""

    def forward(self, pred, mask):
        return structure_loss(pred, mask)
