"""
losses/lovasz.py
────────────────────────────────────────────────────────────────
Lovász-Hinge Loss — a tractable convex surrogate for optimising the
Jaccard index (IoU) directly, instead of a proxy like BCE or Dice.

Reference
─────────
Berman, Triki, Blaschko — "The Lovász-Softmax loss: A tractable
surrogate for the optimization of the intersection-over-union
measure in neural networks", CVPR 2018.
https://arxiv.org/abs/1705.08790

Why use it
──────────
  • IoU/Jaccard is the metric segmentation is usually *evaluated*
    on, but it's a set-based, non-differentiable quantity — can't
    backprop through it directly.
  • The Lovász extension of the Jaccard *loss* (1 - IoU) is convex
    and piecewise linear in the errors, so it can be optimised with
    a standard hinge-style surrogate on the sorted per-pixel errors.
  • In practice it tends to sharpen boundaries and fix small/hard
    regions better than BCE/Dice alone, since gradient weight is
    concentrated on the pixels that actually move the IoU.

Usage notes
───────────
  • `pred` are raw logits (no sigmoid applied), `mask` is 0/1.
  • `per_image=True` (default) computes the loss per sample in the
    batch and averages — this matches the reference implementation
    and behaves better than flattening the whole batch at once.
  • Like Tversky/Focal above, this is usually combined with BCE
    early in training rather than used alone from scratch, since
    the sorted-error gradient can be noisy on an untrained model:
        loss = BCE + λ * LovaszHinge
  • `ignore` lets you mask out a label value (e.g. -1) that should
    not contribute to the loss, same convention as ignore_index.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Gradient of the Lovász extension w.r.t. sorted errors.

    Computes, at each prefix length k of the error-sorted pixel list,
    how much the Jaccard *loss* changes if that prefix is treated as
    the "wrong" set — i.e. the marginal IoU cost of each additional
    mistake, most-confident-mistake first.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_binary(scores: torch.Tensor, labels: torch.Tensor,
                     ignore: float = None):
    """Flatten predictions/labels to 1D, optionally dropping `ignore`."""
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    return scores[valid], labels[valid]


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Core Lovász hinge on a single flattened (logits, labels) pair.

    signs   = ±1 encoding of the label
    errors  = 1 - logits * signs   (hinge-style margin error per pixel)
    Sort errors descending (worst mistakes first), weight the ReLU'd
    errors by the Lovász gradient of the sorted ground truth, sum.
    """
    if len(labels) == 0:
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm.data]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge_loss(pred: torch.Tensor, mask: torch.Tensor,
                       per_image: bool = True, ignore: float = None) -> torch.Tensor:
    """
    Lovász-Hinge loss for binary segmentation.

    pred : logits, shape [B,1,H,W] (or [B,H,W])
    mask : 0/1 ground truth, same shape as pred
    per_image : compute per-sample and average (recommended) vs.
                flattening the whole batch into one long vector.
    """
    if per_image:
        losses = []
        for logit, lab in zip(pred, mask):
            flat_logit, flat_lab = _flatten_binary(logit, lab, ignore)
            losses.append(_lovasz_hinge_flat(flat_logit, flat_lab))
        return torch.stack(losses).mean()
    else:
        flat_logit, flat_lab = _flatten_binary(pred, mask, ignore)
        return _lovasz_hinge_flat(flat_logit, flat_lab)


@register_loss("lovasz")
@register_loss("lovasz_hinge")
class LovaszHingeLoss(nn.Module):
    """Lovász-Hinge loss — direct IoU surrogate for binary segmentation."""

    def __init__(self, per_image: bool = True, ignore: float = None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, pred, mask):
        return lovasz_hinge_loss(pred, mask, per_image=self.per_image, ignore=self.ignore)


@register_loss("bce_lovasz")
class BCELovaszLoss(nn.Module):
    """BCE + Lovász-Hinge combo — stabilises early training, sharpens IoU later."""

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, pred, mask):
        bce = F.binary_cross_entropy_with_logits(pred, mask)
        lov = lovasz_hinge_loss(pred, mask)
        return bce + self.lam * lov
