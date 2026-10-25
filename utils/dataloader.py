"""
utils/dataloader.py
────────────────────────────────────────────────────────────────
Fixes vs previous version:
  1. [AUGMENTATION] Replaced torchvision spatial augmentation + manual
     seed-sync hack with albumentations pipeline.
     albumentations handles image+mask jointly natively, so the
     random.seed / torch.manual_seed dance before every transform is
     no longer needed and has been removed.
  2. [AUGMENTATION] Added stronger augmentation tier that matches what
     2024–2025 polyp segmentation papers actually use:
       - ElasticTransform      (deforms polyp shape realistically)
       - GridDistortion        (simulates bowel wall curvature)
       - OpticalDistortion     (endoscope lens distortion)
       - CoarseDropout         (occlusion / specular highlight simulation)
       - GaussNoise            (sensor noise)
     All spatial ops are applied to image AND mask simultaneously.
     Color/noise ops are applied to image only (albumentations default).
  3. [AUGMENTATION] augmentation parameter now accepts three tiers:
       True / 'strong'  → full albumentations pipeline  (training)
       'basic'          → flip + rotation only           (light experiments)
       False            → resize only                    (test/val)
  4. [MINOR] filter_files unchanged; get_loader signature unchanged.

Dependencies:
    pip install albumentations>=1.4
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False

from utils.seed import get_worker_init_fn, get_generator


# ─────────────────────────────────────────────────────────────────────────────
# Albumentations pipeline builders
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
FIXED_IMAGE_SIZE = 352


def resolve_hw(size, width=None):
    return FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE


def _build_strong_aug(trainsize: int) -> "A.Compose":
    """
    Full augmentation pipeline used by 2024–2025 SOTA polyp papers.

    Spatial transforms are listed first so they are applied to both
    image and mask. Color/noise transforms only affect the image
    (albumentations skips them for masks automatically).

    References:
      - NanoNetB + CRF/TTA (Frontiers Robotics AI, 2024)
        RandomRotate90, Crop, VFlip, ElasticTransform, GridDistortion,
        OpticalDistortion, HFlip, GaussNoise, CoarseDropout
      - Polyp augmentation study (IJCARS, 2020)
        Elastic deformation improves Kvasir-SEG (+1.7 Dice pts).
    """
    height, width = resolve_hw(trainsize)
    return A.Compose([
        # ── Spatial (applied to image + mask jointly) ─────────────────
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
            border_mode=0, p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=(-0.2, 0.2), p=1.0),
        ], p=0.4),
        A.Resize(height, width),
        # ── Image-only (mask is left unchanged) ───────────────────────
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0, p=0.3),
        # ── Normalize + to tensor ─────────────────────────────────────
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_basic_aug(trainsize: int) -> "A.Compose":
    """Lightweight pipeline: flip + rotation only."""
    height, width = resolve_hw(trainsize)
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height, width),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_val_pipeline(trainsize: int) -> "A.Compose":
    """No augmentation — resize + normalize only."""
    height, width = resolve_hw(trainsize)
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Fallback torchvision pipeline (used only when albumentations is not installed)
# ─────────────────────────────────────────────────────────────────────────────

def _torchvision_strong(trainsize):
    """
    Legacy pipeline kept for environments without albumentations.
    Uses the old seed-sync trick for spatial consistency.
    """
    height, width = resolve_hw(trainsize)
    img_tf = T.Compose([
        T.RandomRotation(90),
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((height, width)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    gt_tf = T.Compose([
        T.RandomRotation(90),
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((height, width)),
        T.ToTensor()])
    return img_tf, gt_tf


def _torchvision_val(trainsize):
    height, width = resolve_hw(trainsize)
    img_tf = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    gt_tf = T.Compose([
        T.Resize((height, width)),
        T.ToTensor()])
    return img_tf, gt_tf


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _list_image_gt_paths(image_root, gt_root):
    images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                     if f.endswith('.jpg') or f.endswith('.png')])
    gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)
                  if f.endswith('.png')])
    assert len(images) == len(gts), (
        f'Image/GT count mismatch: {len(images)} vs {len(gts)}')

    filtered_images, filtered_gts = [], []
    for img_path, gt_path in zip(images, gts):
        with Image.open(img_path) as img, Image.open(gt_path) as gt:
            if img.size == gt.size:
                filtered_images.append(img_path)
                filtered_gts.append(gt_path)
    return filtered_images, filtered_gts


def split_train_val_paths(image_root, gt_root, val_ratio=0.2, split_seed=42):
    images, gts = _list_image_gt_paths(image_root, gt_root)
    n = len(images)
    if n < 2:
        raise ValueError(f'Need at least 2 samples for train/val split, got {n}')

    val_count = int(round(n * val_ratio))
    val_count = min(max(1, val_count), n - 1)

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n).tolist()
    val_indices = sorted(perm[:val_count])
    train_indices = sorted(perm[val_count:])

    train_images = [images[i] for i in train_indices]
    train_gts = [gts[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_gts = [gts[i] for i in val_indices]

    info = {
        'split_seed': int(split_seed),
        'val_ratio': float(val_ratio),
        'total_size': int(n),
        'train_size': int(len(train_images)),
        'val_size': int(len(val_images)),
        'train_images': [os.path.basename(p) for p in train_images],
        'val_images': [os.path.basename(p) for p in val_images],
    }
    return train_images, train_gts, val_images, val_gts, info


class PolypDataset(data.Dataset):
    """
    augmentations accepts:
      True / 'strong'  → full albumentations pipeline
      'basic'          → flip + rotation only
      False / None     → resize only (val/test mode)
    """

    def __init__(self, image_root, gt_root, trainsize, augmentations=True,
                 image_paths=None, gt_paths=None):
        self.trainsize     = FIXED_IMAGE_SIZE
        self.augmentations = augmentations

        if image_paths is None or gt_paths is None:
            self.images, self.gts = _list_image_gt_paths(image_root, gt_root)
        else:
            assert len(image_paths) == len(gt_paths), (
                f'Image/GT count mismatch: {len(image_paths)} vs {len(gt_paths)}')
            self.images = list(image_paths)
            self.gts = list(gt_paths)
        self.size = len(self.images)

        # ── Choose pipeline ───────────────────────────────────────────
        use_aug  = augmentations not in (False, None, 'none')
        use_full = augmentations in (True, 'strong')

        if _ALBUMENTATIONS_AVAILABLE:
            self._use_albu = True
            if use_full:
                self._albu = _build_strong_aug(trainsize)
            elif use_aug:          # 'basic'
                self._albu = _build_basic_aug(trainsize)
            else:
                self._albu = _build_val_pipeline(trainsize)
            self._img_transform = None
            self._gt_transform  = None
        else:
            # Fallback to torchvision (with legacy seed-sync)
            import warnings
            warnings.warn(
                "albumentations not found — falling back to torchvision. "
                "Install with: pip install albumentations>=1.4",
                UserWarning, stacklevel=2)
            self._use_albu = False
            if use_aug:
                self._img_transform, self._gt_transform = \
                    _torchvision_strong(trainsize)
            else:
                self._img_transform, self._gt_transform = \
                    _torchvision_val(trainsize)

    # ── Item loading ──────────────────────────────────────────────────────────

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt    = self.binary_loader(self.gts[index])

        if self._use_albu:
            # albumentations expects numpy arrays
            image_np = np.array(image)                   # H×W×3  uint8
            gt_np    = np.array(gt)                      # H×W    uint8

            result   = self._albu(image=image_np, mask=gt_np)
            image_t  = result['image']                   # C×H×W  float32 tensor
            gt_t     = result['mask'].unsqueeze(0).float() / 255.0
            return image_t, gt_t
        else:
            # Legacy torchvision path with seed synchronisation
            seed = np.random.randint(2147483647)
            random.seed(seed); torch.manual_seed(seed)
            if self._img_transform is not None:
                image = self._img_transform(image)
            random.seed(seed); torch.manual_seed(seed)
            if self._gt_transform is not None:
                gt = self._gt_transform(gt)
            return image, gt

    # ── Helpers ───────────────────────────────────────────────────────────────

    def filter_files(self):
        assert len(self.images) == len(self.gts), (
            f'Image/GT count mismatch: {len(self.images)} vs {len(self.gts)}')
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            with Image.open(img_path) as img, Image.open(gt_path) as gt:
                if img.size == gt.size:
                    images.append(img_path)
                    gts.append(gt_path)
        self.images = images
        self.gts    = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# Test dataset — unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = resolve_hw(testsize)
        self.images = sorted([image_root + f for f in os.listdir(image_root)
                              if f.endswith('.jpg') or f.endswith('.png')])
        self.gts    = sorted([gt_root    + f for f in os.listdir(gt_root)
                              if f.endswith('.tif') or f.endswith('.png')])
        self.transform = T.Compose([
            T.Resize(self.testsize),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
        self.gt_transform = T.ToTensor()
        self.size  = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt    = self.binary_loader(self.gts[self.index])
        name  = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory — signature unchanged
# ─────────────────────────────────────────────────────────────────────────────

def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=None, pin_memory=True,
               augmentation=True, seed=42):
    """
    Build a reproducible DataLoader for training.

    augmentation:
      True / 'strong'  → full albumentations pipeline (default)
      'basic'          → flip + rotation only
      False            → no augmentation

    num_workers: defaults to min(4, cpu_count) when None.
    """
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=get_worker_init_fn(),
        generator=get_generator(seed),
        persistent_workers=(num_workers > 0),
    )
    return data_loader


def get_train_val_loaders(image_root, gt_root, batchsize, trainsize,
                          val_ratio=0.2, split_seed=42,
                          shuffle=True, num_workers=None, pin_memory=True,
                          augmentation=True, seed=42):
    """
    Build deterministic 80/20-style train/val loaders from TrainDataset.

    The split seed is intentionally separate from the training seed so
    multi-seed experiments compare models on the same validation subset.
    """
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    train_images, train_gts, val_images, val_gts, split_info = split_train_val_paths(
        image_root=image_root,
        gt_root=gt_root,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )

    train_dataset = PolypDataset(
        image_root, gt_root, trainsize, augmentation,
        image_paths=train_images, gt_paths=train_gts,
    )
    val_dataset = PolypDataset(
        image_root, gt_root, trainsize, False,
        image_paths=val_images, gt_paths=val_gts,
    )

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=get_worker_init_fn(),
        generator=get_generator(seed),
        persistent_workers=(num_workers > 0),
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=get_worker_init_fn(),
        generator=get_generator(split_seed),
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, split_info
