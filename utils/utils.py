"""
utils/utils.py
────────────────────────────────────────────────────────────────
Fix vs original:
  AvgMeter.losses grew unboundedly — each update() appended one tensor and
  nothing ever trimmed the list.  Over 100 epochs × steps this accumulates
  thousands of GPU tensors in CPU memory.  Fixed: keep only the last `num`
  (default 40) entries, matching the existing show() window.
"""

import os
import random
from copy import deepcopy
import numpy as np
import torch
from thop import profile, clever_format

from utils.seed import seed_everything, seed_epoch, log_versions


def set_seed(seed: int) -> None:
    """Alias for seed_everything() — kept for backward compatibility."""
    seed_everything(seed)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=50):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay


class AvgMeter:
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val    = 0
        self.avg    = 0
        self.sum    = 0
        self.count  = 0
        self.losses = []   # rolling window — capped at self.num entries

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count
        self.losses.append(val)
        # FIX: trim to sliding window so the list never grows unboundedly.
        # Original code appended forever; after 100 epochs this accumulated
        # thousands of detached tensors in CPU memory.
        if len(self.losses) > self.num:
            self.losses = self.losses[-self.num:]

    def show(self):
        return torch.mean(torch.stack(self.losses))


def split_model_output(out):
    """
    Normalize a model's forward() output into (main, aux):
      - single-tensor models (e.g. HarDMSEG)           -> (out, [])
      - tuple/list models   (e.g. PraNet: out[0]=main)  -> (out[0], list(out[1:]))
    """
    if isinstance(out, (list, tuple)):
        return out[0], list(out[1:])
    return out, []


def fuse_outputs(main, aux):
    """
    Combine (main, aux) into the single map used for eval/inference metrics.

    - No aux (single-output models, e.g. HarDMSEG): just `main`.
    - Exactly one aux output (dual-branch fusion architectures whose paper
      defines the final prediction as the sum of two decoder heads, e.g.
      Polyp-PVT-style P1+P2): `main + aux[0]`.
    - More than one aux output (deep-supervision architectures, e.g. PraNet's
      lateral_map_5/4/3): those extra maps exist purely to supervise
      intermediate decoder stages during training. The original paper only
      ever reads the single finest map at inference (no summation — see
      PraNet's MyTest.py, which uses `res2` alone), so we return `main` alone.
    """
    if len(aux) == 1:
        return main + aux[0]
    return main


def compute_loss_with_aux(criterion, main, aux, masks):
    """L = L_main + sum(L_aux_i), equal weights — matches deep-supervision
    training recipes (e.g. PraNet's loss2 + loss3 + loss4 + loss5)."""
    loss = criterion(main, masks)
    for a in aux:
        loss = loss + criterion(a, masks)
    return loss


def cal_params(model, input_size=(1, 3, 352, 352)):
    device = next(model.parameters()).device
    profile_model = deepcopy(model).to(device)
    dummy = torch.randn(input_size).to(device)
    profile_model.eval()
    with torch.no_grad():
        flops, params = profile(profile_model, inputs=(dummy,), verbose=False)
    del profile_model
    flops, params = clever_format([flops, params], '%.3f')
    return flops, params
