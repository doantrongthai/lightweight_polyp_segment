"""
losses/dna_tloss.py
────────────────────────────────────────────────────────────────
Dynamic-Nu T-Loss (DNA-TLoss) for robust colonoscopy polyp segmentation
(Norouziazad et al., 2026, "Robust colonoscopy polyp segmentation using
dynamic-Nu T-Loss with multi-scale and uncertainty-aware adaptation",
Front. Med. 12:1657123).

Extends the Student's-t negative log-likelihood loss (T-Loss, Gonzalez-
Jimenez et al., MICCAI 2023) with three components (Section 2.3):

  1. Adaptive ν — a lightweight NuPredictor CNN + 2-layer MLP predicts a
     single per-image degrees-of-freedom scalar ν (Softplus > 0). Low ν
     gives heavy tails (robust to outliers/noisy labels); high ν gives
     Gaussian-like behaviour (sensitive to errors).
  2. Per-pixel precision map λ — a learnable log-precision map per scale,
     zero-initialised (= unit variance), trained jointly with the model.
  3. Multi-scale aggregation — the loss is computed at scales
     {1.0, 0.5, 0.25} with weights {1.0, 0.5, 0.3} (Eq. 4) and summed.

For one scale s, with residual δ_i = p_i − y_i and log-precision λ_i,
the scale loss is the batched Student-t NLL (Eq. 3), derived from the
Student-t density with variance e^{-λ_i}:

    L_s = mean_i[ -lgamma((ν+1)/2) + lgamma(ν/2) + 0.5·log(πν)
                   - 0.5·λ_i + (ν+1)/2 · log(1 + δ_i²·e^{λ_i}/ν) ]

Hyperparameter choices (from the paper)
────────────────────────────────────────
  scales  = (1.0, 0.5, 0.25)   weights = (1.0, 0.5, 0.3)   (Section 2.3.2)
  NuPredictor: 3× (3×3 conv, stride 1, pad 1, ReLU) → AdaptiveAvgPool2d(1)
               → MLP(hidden=32) → Softplus
  λ maps: nn.Parameter, zero-initialised, one per scale, shape (1,1,h_s,w_s)
  eps = 1e-6 — numerical stability only (added inside log/exp, per paper).

Note on inputs
───────────────
Unlike the other losses in this package, DNA-TLoss owns learnable
parameters (the NuPredictor and the λ-maps), so — as in the paper, where
"all loss parameters are updated by backpropagation along with the
network weights" — it must stay instantiated across training (same
object registered in the optimizer's param groups) rather than being
re-created every call. `base_size=(H, W)` should match the resolution
the segmentation network trains at (512×608 in the paper), since the
λ-maps are pre-allocated for that resolution's scaled versions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_loss

_EPS = 1e-6


class NuPredictor(nn.Module):
    """Predicts a per-image degrees-of-freedom ν from image features."""

    def __init__(self, in_channels: int = 1, feat_channels: int = 16, hidden_dim: int = 32):
        super().__init__()
        # Thay mẹ LazyConv2d bằng Conv2d để né lỗi lệch device của PyTorch
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensures nu > 0 (and never exactly 0 -> stability)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> nu: (B, 1, 1, 1), strictly positive."""
        feat = self.pool(self.conv(x)).flatten(1)   # (B, feat_channels)
        nu = self.mlp(feat)                          # (B, 1)
        return nu.view(-1, 1, 1, 1) + _EPS


def student_t_nll(residual: torch.Tensor, log_precision: torch.Tensor,
                   nu: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    Per-pixel Student's-t negative log-likelihood for one scale (Eq. 3).
    """
    const = (-torch.lgamma((nu + 1.0) / 2.0) + torch.lgamma(nu / 2.0)
             + 0.5 * torch.log(math.pi * nu + eps))
    quad = ((nu + 1.0) / 2.0) * torch.log(
        1.0 + (residual ** 2) * torch.exp(log_precision) / (nu + eps) + eps
    )
    per_pixel = const - 0.5 * log_precision + quad
    return per_pixel.mean()


@register_loss("dna_tloss")
@register_loss("dnatloss")
class DNATLoss(nn.Module):
    """
    Multi-scale dynamic-Nu T-Loss (DNA-TLoss), combining adaptive ν,
    per-pixel precision maps λ, and multi-scale aggregation (Eq. 4).
    """

    def __init__(self, in_channels=1, base_size=(512, 608),
                 scales=(1.0, 0.5, 0.25), scale_weights=(1.0, 0.5, 0.3)):
        super().__init__()
        assert len(scales) == len(scale_weights), "scales/weights length mismatch"
        self.scales = scales
        self.scale_weights = scale_weights

        # Lightweight NuPredictor: shared across scales, one ν per image.
        self.nu_predictor = NuPredictor(in_channels=in_channels)

        # One learnable, zero-initialised log-precision map per scale.
        H, W = base_size
        self.log_precision = nn.ParameterDict()
        for s in scales:
            h, w = max(1, round(H * s)), max(1, round(W * s))
            self.log_precision[self._scale_key(s)] = nn.Parameter(
                torch.zeros(1, 1, h, w)
            )

    @staticmethod
    def _scale_key(s: float) -> str:
        return f"s{str(s).replace('.', '_')}"

    def forward(self, pred: torch.Tensor, mask: torch.Tensor,
                image: torch.Tensor = None) -> torch.Tensor:
        
        # Cho chốt luôn hàm loss này cùng device với tensor truyền vào
        if next(self.parameters()).device != pred.device:
            self.to(pred.device)

        prob = torch.sigmoid(pred)
        nu_input = image if image is not None else prob.detach()
        nu = self.nu_predictor(nu_input)  # (B, 1, 1, 1), per-image

        total = pred.new_zeros(())
        for s, w in zip(self.scales, self.scale_weights):
            lam = self.log_precision[self._scale_key(s)]
            h, w_ = lam.shape[-2], lam.shape[-1]

            # Bilinear for the probability map, nearest for the binary mask
            p_s = F.interpolate(prob, size=(h, w_), mode="bilinear", align_corners=False)
            y_s = F.interpolate(mask, size=(h, w_), mode="nearest")

            residual = p_s - y_s
            total = total + w * student_t_nll(residual, lam, nu)

        return total