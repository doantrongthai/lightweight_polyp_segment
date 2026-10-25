"""
losses/boundary.py
────────────────────────────────────────────────────────────────
Boundary Loss for binary segmentation, adapted from:

    Kervadec et al., "Boundary loss for highly unbalanced segmentation"
    (MIDL 2019 / MedIA 2021)
    Official repo: https://github.com/LIVIAETS/boundary-loss

Original formulation (`SurfaceLoss` in the paper's `losses.py`)
──────────────────────────────────────────────────────────────
The GT mask is first converted to a *signed distance map* `dist_map`
via a Euclidean distance transform (EDT):

    dist_map = EDT(background) * background_mask
               - (EDT(foreground) - 1) * foreground_mask

so pixels are:
  • positive  and growing with distance, outside the object
  • negative  and growing (in magnitude) toward the object's centre,
    inside the object
  • ~0 right at the boundary

The loss itself is then just a dot product between the predicted
probabilities and this pre-computed distance map:

    L_boundary = mean( softmax_probs * dist_map )

Minimising this pushes probability mass away from pixels far outside
the object and rewards probability mass near/inside it, with the
gradient's magnitude increasing with distance to the boundary —
hence "boundary loss": it directly supervises the *shape* of the
prediction, unlike pixel-wise losses like BCE/Dice which are blind to
geometry.

Adaptation notes for this codebase
───────────────────────────────────
The original repo works with multi-class one-hot tensors
`[B, K, H, W]` and precomputes `dist_maps` once per sample in the
dataloader (since scipy's EDT is CPU-only and not differentiable
w.r.t. the mask). Here we:
  • target the binary single-channel case (`pred`, `mask` both
    `[B, 1, H, W]`) to match `bce.py` / `structure.py`;
  • compute the distance map on-the-fly inside `forward`, under
    `torch.no_grad()`, straight from the GT `mask` tensor — no
    dataloader changes required, at the cost of a per-step CPU
    round-trip (scipy `distance_transform_edt`);
  • gate on `mask.any()` per-sample exactly like `one_hot2dist`, so
    blank (empty) masks contribute a zero distance map instead of
    crashing/propagating garbage;
  • use predicted **probabilities** (`sigmoid(pred)`), the binary
    equivalent of the original's softmax probabilities.

Known caveat (documented by the original authors): boundary loss is
unstable when used alone from scratch, since the distance map alone
gives no signal for a totally-empty prediction. The upstream repo
combines it with a region-based term (e.g. Generalized Dice) and
linearly ramps its weight up over training. We expose that as
`DiceBoundaryLoss` below, mirroring how `tversky_focal.py` exposes
both the combined and single-term ablations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance

from . import register_loss


@torch.no_grad()
def mask_to_dist_map(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a binary GT mask [B, 1, H, W] into a signed Euclidean
    distance map of the same shape, following `one_hot2dist` from the
    original boundary-loss repo (binary, single-channel case).

    dist_map < 0 inside the foreground object, > 0 outside, ~0 at the
    boundary. Computed per-sample on CPU via scipy; not differentiable
    (and doesn't need to be — only `pred` needs gradients).
    """
    device = mask.device
    mask_np = mask.detach().cpu().numpy().astype(bool)  # [B, 1, H, W]
    B = mask_np.shape[0]

    dist_np = np.zeros_like(mask_np, dtype=np.float32)
    for b in range(B):
        posmask = mask_np[b, 0]
        if posmask.any():
            negmask = ~posmask
            dist_np[b, 0] = (
                eucl_distance(negmask) * negmask
                - (eucl_distance(posmask) - 1) * posmask
            )
        # else: leave as zeros (empty mask -> no boundary to supervise)

    return torch.from_numpy(dist_np).to(device=device, dtype=torch.float32)


def boundary_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    L = mean( sigmoid(pred) * signed_dist_map(mask) )

    Binary counterpart of `SurfaceLoss.__call__` in the original repo.
    """
    dist_map = mask_to_dist_map(mask)
    prob = torch.sigmoid(pred)
    return (prob * dist_map).mean()


def dice_loss(pred: torch.Tensor, mask: torch.Tensor,
              smooth: float = 1e-8) -> torch.Tensor:
    """Plain soft Dice, used as the stabilising region term (as in the
    upstream repo, which pairs SurfaceLoss with GeneralizedDice)."""
    prob = torch.sigmoid(pred)
    inter = (prob * mask).sum(dim=(1, 2, 3))
    union = (prob + mask).sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return (1.0 - dice).mean()


@register_loss("boundary")
class BoundaryLoss(nn.Module):
    """Boundary-only ablation (unstable alone; kept for comparison)."""

    def forward(self, pred, mask):
        return boundary_loss(pred, mask)


@register_loss("dice_boundary")
class DiceBoundaryLoss(nn.Module):
    """
    Dice + Boundary, with a linearly increasing boundary weight —
    mirrors the alpha-scheduling trick from the original repo/paper
    (`alpha` starts at 0.01 and grows by +0.01/epoch there).

    Call `.step_epoch()` once per epoch from the training loop to
    advance the schedule; `boundary_weight` is clamped to
    `max_boundary_weight`.
    """

    def __init__(self, max_boundary_weight: float = 0.5,
                 weight_step: float = 0.01):
        super().__init__()
        self.max_boundary_weight = max_boundary_weight
        self.weight_step = weight_step
        self.boundary_weight = 0.01

    def step_epoch(self):
        self.boundary_weight = min(
            self.boundary_weight + self.weight_step, self.max_boundary_weight
        )

    def forward(self, pred, mask):
        return (
            (1.0 - self.boundary_weight) * dice_loss(pred, mask)
            + self.boundary_weight * boundary_loss(pred, mask)
        )
