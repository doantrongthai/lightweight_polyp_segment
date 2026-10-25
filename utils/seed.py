"""
utils/seed.py
─────────────────────────────────────────────────────────────────────────────
Centralised reproducibility utilities for the Polyp-PVT pipeline.

Fix vs original:
  torch.set_num_threads(1) and torch.set_num_interop_threads(1) have been
  REMOVED from seed_everything().  On T4/Colab they throttle all intra-op
  parallelism, slowing CPU-side preprocessing by up to 4×.  The dominant
  source of non-determinism they guarded against (CPU matmul thread ordering)
  is already handled by the cuDNN/CUDA flags below.  OMP_NUM_THREADS and
  MKL_NUM_THREADS are kept at 1 to stabilise numpy inside DataLoader workers.

Sources of non-determinism addressed
──────────────────────────────────────
 1. Environment variables         (must precede any CUDA / MKL / OMP init)
 2. Python built-in RNG
 3. NumPy RNG
 4. PyTorch CPU + CUDA RNGs
 5. cuDNN: algorithm benchmarking + deterministic kernel selection
 6. TF32 (Ampere+ GPUs)           — disabled unconditionally
 7. Float32 matmul precision      — set to "highest"
 8. torch.use_deterministic_algorithms
 9. fill_uninitialized_memory     — guards torch.empty / resize_
10. SDPA / Flash Attention        — force math backend
11. DataLoader workers            — worker_init_fn + seeded Generator
"""

import os
import random

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Primary API
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """
    Eliminate every controllable source of non-determinism.
    Call once at the very start of each run / seed iteration,
    BEFORE model construction, DataLoader creation, and training.
    """

    # ── 1. Environment variables ──────────────────────────────────────────
    os.environ["PYTHONHASHSEED"]          = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Keep numpy/scipy in workers deterministic without killing throughput.
    # Note: set_num_threads(1) removed — see module docstring.
    os.environ["OMP_NUM_THREADS"]         = "1"
    os.environ["MKL_NUM_THREADS"]         = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    # ── 2. Python RNG ─────────────────────────────────────────────────────
    random.seed(seed)

    # ── 3. NumPy RNG ──────────────────────────────────────────────────────
    np.random.seed(seed)

    # ── 4. PyTorch CPU + CUDA RNGs ────────────────────────────────────────
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── 5. cuDNN ──────────────────────────────────────────────────────────
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ── 6. TF32 ───────────────────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False

    # ── 7. Float32 matmul precision ───────────────────────────────────────
    torch.set_float32_matmul_precision("highest")

    # ── 8. Deterministic algorithm enforcement ────────────────────────────
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # ── 9. fill_uninitialized_memory ──────────────────────────────────────
    if hasattr(torch.utils, "deterministic"):
        torch.utils.deterministic.fill_uninitialized_memory = True  # type: ignore[attr-defined]

    # ── 10. SDPA / Flash Attention backends ───────────────────────────────
    if hasattr(torch.backends, "cuda"):
        _set_sdpa_deterministic()


def _set_sdpa_deterministic() -> None:
    """Force SDPA to use the deterministic math backend."""
    for flag, value in [
        ("enable_flash_sdp",         False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp",          True),
        ("enable_cudnn_sdp",         False),
    ]:
        if hasattr(torch.backends.cuda, flag):
            getattr(torch.backends.cuda, flag)(value)


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch lightweight re-seed
# ─────────────────────────────────────────────────────────────────────────────

def seed_epoch(base_seed: int, epoch: int) -> None:
    """
    Lightweight per-epoch re-seed for data augmentation reproducibility
    (RandomRotation, ColorJitter, etc. in PolypDataset.__getitem__).

    Call at the START of every training epoch in the training loop.
    Each (base_seed, epoch) pair is unique so augmentation sequences
    differ across epochs but are fixed across runs.
    """
    epoch_seed = (base_seed * 1000 + epoch) % (2 ** 32)
    random.seed(epoch_seed)
    np.random.seed(epoch_seed)
    torch.manual_seed(epoch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(epoch_seed)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_worker_init_fn():
    """
    Return a worker_init_fn that properly seeds each DataLoader worker.

    PyTorch DataLoader spawns `num_workers` subprocesses; each receives a
    unique torch seed derived from the main seed + worker_id.  Without this
    function, NumPy and Python's random module inside workers are NOT seeded,
    causing non-reproducible augmentation (flip, rotation, colour jitter).
    """
    def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
        worker_seed = torch.initial_seed() % (2 ** 32)
        os.environ["PYTHONHASHSEED"] = str(worker_seed)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    return _worker_init_fn


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Return a seeded torch.Generator for DataLoader shuffle reproducibility.

    The DataLoader's internal shuffle generator is decoupled from the global
    RNG state.  Without an explicit generator, shuffling order changes even
    when the global seed is fixed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Environment / version logging
# ─────────────────────────────────────────────────────────────────────────────

def log_versions() -> None:
    """Print library versions and GPU info at the start of every run."""
    import sys
    import platform

    lines = [
        "=" * 64,
        "  ENVIRONMENT",
        "=" * 64,
        f"  Python      : {sys.version.split()[0]}",
        f"  Platform    : {platform.platform()}",
        f"  PyTorch     : {torch.__version__}",
    ]
    for pkg in ("torchvision", "timm", "numpy", "sklearn", "scipy", "thop"):
        try:
            mod = __import__(pkg)
            lines.append(f"  {pkg:<12}: {mod.__version__}")
        except ImportError:
            pass

    if torch.cuda.is_available():
        lines += [
            f"  CUDA        : {torch.version.cuda}",
            f"  cuDNN       : {torch.backends.cudnn.version()}",
            f"  GPU         : {torch.cuda.get_device_name(0)}",
            f"  GPU count   : {torch.cuda.device_count()}",
        ]
    else:
        lines.append("  GPU         : not available (CPU-only run)")

    lines.append("=" * 64)
    print("\n".join(lines))