import torch.nn as nn

from . import register_loss


@register_loss("bce")
class BCELoss(nn.Module):
    """Plain BCE baseline."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, mask):
        return self.loss(pred, mask)
