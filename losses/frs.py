"""
losses/frs.py
────────────────────────────────────────────────────────────────
Fuzzy Rough Set (FRS) loss for medical image segmentation
(Dar & Ganivada, "Fuzzy rough set loss for deep learning-based precise
medical image segmentation", Computerized Medical Imaging and
Graphics 128 (2026) 102716).

The FRS loss combines two components in a convex combination:

  1. Fuzzy similarity loss (L_fuzzy) — pixel-wise fuzzy membership /
     similarity between prediction and ground truth. It captures
     gradual transitions at lesion boundaries (Eqs. 10-13).

  2. Boundary uncertainty model (L_RSA) — fuzzy lower/upper
     approximations (fuzzy-rough-set-inspired) weighted by a
     prediction-uncertainty term. It handles boundary ambiguity and
     mitigates class imbalance (Eqs. 14-21).

        L_FRS = alpha * L_fuzzy + (1 - alpha) * L_RSA        (Eq. 22)

Implementation notes
─────────────────────
  • Eq. 10 defines class membership M_k(i) from the distance between a
    pixel's *feature vector* f_i and the class mean mu_k in feature
    space. A stand-alone loss module (forward(pred, mask), like the
    other losses in this package) only has access to the network's
    final logits, not its intermediate features. We therefore treat
    the predicted foreground probability p_i as a 1-D "feature" and
    estimate mu_k, sigma_k^2 per image over the foreground/background
    populations defined by the ground-truth mask. This keeps the loss
    fully differentiable and preserves the intent of Eq. 10 without
    requiring backbone feature maps.
  • w (Eq. 10) is the trainable, strictly-positive fuzziness parameter.
    It is learned jointly with the network, but — unlike a plain
    softplus parameterisation — it is squashed into a *bounded* range
    [W_MIN, W_MAX] via a sigmoid. An unbounded w is free to grow
    without limit during training (nothing in the loss penalises large
    w), and once w gets large, exp(w)-type terms overflow to inf; the
    backward pass through a hand-written `1/(1+exp(x))` then hits the
    indeterminate form inf/inf (or 0*inf), producing NaN gradients.
    Bounding w removes this failure mode at its source.
  • Every place the paper's sigmoid-shaped functions (Eqs. 10, 12) are
    evaluated, we use `torch.sigmoid` instead of the literal
    `1/(1+exp(x))` formula. Mathematically identical, but
    `torch.sigmoid` is numerically stable for large |x| (PyTorch
    computes its gradient analytically from the already-saturated
    output, avoiding exp() overflow inside the backward pass).
  • alpha = 0.3 is used by default — the value the authors report as
    working best across all five datasets (Section 3.3 / Section 5.7).
  • As noted by the authors themselves (Section 3.2.3), the lower/upper
    approximations (min/max) are a differentiable relaxation of the
    classical rough-set approximations, not a literal implementation
    of indiscernibility-relation-based rough sets.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss

_EPS = 1e-6          # numerical floor (variance, counts, etc.)
_EXP_CLAMP = 20.0    # exp(20) ~ 4.85e8, safely representable in fp32
_W_MIN, _W_MAX = 0.1, 5.0   # bounds enforced on the trainable w


def _inv_sigmoid(p: float) -> float:
    """Raw (pre-sigmoid) value that makes sigmoid(raw) == p, 0<p<1."""
    return math.log(p / (1.0 - p))


def _class_stats(prob: torch.Tensor, mask: torch.Tensor):
    """
    Per-image mean/variance of the predicted probability, split into
    foreground (mask=1) and background (mask=0) populations.
    Returns (mu_fg, var_fg), each broadcastable to [B,1,H,W].

    The variance floor is set to a value comparable to the natural
    scale of `prob` (which lives in [0, 1]) rather than an infinitesimal
    epsilon, so that 1/var_fg cannot blow up to astronomical values
    when a batch happens to contain very few / very uniform
    foreground pixels.
    """
    fg_w = mask
    n_fg = fg_w.sum(dim=(1, 2, 3), keepdim=True).clamp_min(_EPS)
    mu_fg = (prob * fg_w).sum(dim=(1, 2, 3), keepdim=True) / n_fg
    var_fg = ((prob - mu_fg) ** 2 * fg_w).sum(dim=(1, 2, 3), keepdim=True) / n_fg
    return mu_fg, var_fg.clamp_min(1e-3)


def fuzzy_similarity_loss(pred: torch.Tensor, mask: torch.Tensor,
                          w: torch.Tensor) -> torch.Tensor:
    """
    Fuzzy similarity loss L_fuzzy (Eqs. 10-13).

    pred : raw logits, [B,1,H,W]
    mask : binary ground truth, [B,1,H,W]
    w    : trainable positive scalar controlling class-boundary fuzziness
    """
    prob = torch.sigmoid(pred)

    mu_fg, var_fg = _class_stats(prob, mask)

    # Eq. 10: membership of each pixel to the foreground class.
    # 1 / (1 + exp(x))  ==  sigmoid(-x), but the latter is numerically
    # stable (no exp() overflow) for large |x|.
    d_fg = (prob - mu_fg) ** 2 / var_fg
    m_fg = torch.sigmoid(-d_fg * w)

    # Eq. 11: influence value lambda_k (foreground class). The exponent
    # is clamped before exp() so a large w cannot overflow to inf and
    # later trigger an inf * 0 -> NaN when a term's own weight is exp'd
    # against a zero numerator.
    influence_w = torch.exp(torch.clamp(m_fg * w, max=_EXP_CLAMP))
    lam = ((prob - mu_fg) ** 2 * influence_w).sum(dim=(1, 2, 3)) / \
          influence_w.sum(dim=(1, 2, 3)).clamp_min(_EPS)
    lam = lam.view(-1, 1, 1, 1)

    # Eq. 12: pixel-wise similarity between prediction and ground truth.
    s_i = torch.sigmoid(-(torch.abs(prob - mask) + lam * w))

    # Eq. 13: fuzzy similarity loss (mean dissimilarity over pixels/batch).
    return 1.0 - s_i.mean(dim=(1, 2, 3)).mean()


def boundary_uncertainty_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Boundary uncertainty model L_RSA (Eqs. 14-21): fuzzy lower/upper
    approximations of a fuzzy rough set, weighted by a membership term
    that emphasises uncertain (near 0.5) predictions.
    """
    prob = torch.sigmoid(pred)                        # Eq. 14

    w_m = 1.0 - torch.abs(prob - 0.5) * 2.0            # Eq. 15

    mu_lower = torch.minimum(prob, mask)               # Eq. 16
    mu_upper = torch.maximum(prob, mask)               # Eq. 17

    l_lower = (mu_lower - mask) ** 2                   # per-pixel term of Eq. 18
    l_upper = (mu_upper - mask) ** 2                   # per-pixel term of Eq. 19

    l_sum = l_lower + l_upper                          # Eq. 20
    l_rsa = (w_m * l_sum).mean(dim=(1, 2, 3))           # Eq. 21

    return l_rsa.mean()


def frs_loss(pred: torch.Tensor, mask: torch.Tensor,
            w: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """Combined FRS loss (Eq. 22): alpha * fuzzy + (1 - alpha) * RSA."""
    l_fuzzy = fuzzy_similarity_loss(pred, mask, w)
    l_rsa = boundary_uncertainty_loss(pred, mask)
    return alpha * l_fuzzy + (1.0 - alpha) * l_rsa


@register_loss("frs")
class FRSLoss(nn.Module):
    """
    Fuzzy Rough Set loss (Dar & Ganivada, 2026).

    alpha : convex-combination weight between the fuzzy similarity term
            and the boundary uncertainty term (Eq. 22). alpha=0.3
            matches the paper's reported optimum (Section 3.3).
    init_w: initial value of the trainable fuzziness parameter w in
            Eq. 10. w is kept inside [_W_MIN, _W_MAX] via a sigmoid
            reparameterisation so it cannot grow unbounded and
            destabilise training (see module docstring).
    """

    def __init__(self, alpha: float = 0.3, init_w: float = 1.0):
        super().__init__()
        self.alpha = alpha
        init_frac = (init_w - _W_MIN) / (_W_MAX - _W_MIN)
        self._raw_w = nn.Parameter(torch.tensor(_inv_sigmoid(init_frac)))

    @property
    def w(self) -> torch.Tensor:
        return _W_MIN + (_W_MAX - _W_MIN) * torch.sigmoid(self._raw_w)

    def forward(self, pred, mask):
        return frs_loss(pred, mask, self.w, self.alpha)


@register_loss("fuzzy_similarity")
class FuzzySimilarityLoss(nn.Module):
    """Fuzzy similarity term only — ablation: 'No rough set approx.' row."""

    def __init__(self, init_w: float = 1.0):
        super().__init__()
        init_frac = (init_w - _W_MIN) / (_W_MAX - _W_MIN)
        self._raw_w = nn.Parameter(torch.tensor(_inv_sigmoid(init_frac)))

    @property
    def w(self) -> torch.Tensor:
        return _W_MIN + (_W_MAX - _W_MIN) * torch.sigmoid(self._raw_w)

    def forward(self, pred, mask):
        return fuzzy_similarity_loss(pred, mask, self.w)


@register_loss("boundary_uncertainty")
@register_loss("rsa")
class BoundaryUncertaintyLoss(nn.Module):
    """Boundary uncertainty model only — ablation: 'No fuzzy similarity' row."""

    def forward(self, pred, mask):
        return boundary_uncertainty_loss(pred, mask)
