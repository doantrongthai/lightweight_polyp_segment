import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.dataloader import TestDataset
from utils.utils import split_model_output, fuse_outputs


def _forward_main(model, image):
    main_out, aux_outputs = split_model_output(model(image))
    return fuse_outputs(main_out, aux_outputs)


def _load_rgb(path, target_shape=None):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if target_shape is not None and image.shape[:2] != target_shape:
        image = cv2.resize(
            image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return image


def draw_prediction_gt_contours(image_rgb, pred_mask, gt_mask):
    """
    Draw outlines only: prediction in green, ground truth in red.
    No filled mask is painted over the image.
    """
    overlay = image_rgb.copy()
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    pred_contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    gt_contours, _ = cv2.findContours(
        gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    thickness = max(2, int(round(max(image_rgb.shape[:2]) / 256)))
    cv2.drawContours(overlay, gt_contours, -1, (255, 0, 0), thickness + 1)
    cv2.drawContours(overlay, pred_contours, -1, (0, 255, 0), thickness)
    return overlay


@torch.no_grad()
def save_prediction_maps(model, test_path, datasets, testsize, device, output_root):
    """
    Save probability maps, binary masks, and original-image contour overlays.
    """
    model.eval()
    saved = {}

    for dataset in datasets:
        data_path = os.path.join(test_path, dataset)
        image_root = os.path.join(data_path, "images", "")
        gt_root = os.path.join(data_path, "masks", "")
        loader = TestDataset(image_root, gt_root, testsize)

        prob_dir = os.path.join(output_root, dataset, "prob")
        binary_dir = os.path.join(output_root, dataset, "binary")
        contour_dir = os.path.join(output_root, dataset, "contours")
        os.makedirs(prob_dir, exist_ok=True)
        os.makedirs(binary_dir, exist_ok=True)
        os.makedirs(contour_dir, exist_ok=True)

        for _ in range(loader.size):
            image, gt, name = loader.load_data()
            gt_np = np.asarray(gt, np.float32)
            gt_bin = (gt_np >= 0.5 * (gt_np.max() + 1e-8)).astype(np.uint8)
            image_path = loader.images[loader.index - 1]
            image = image.to(device)

            main = _forward_main(model, image)
            prob = F.interpolate(
                main, size=gt_np.shape, mode="bilinear", align_corners=False
            )
            prob = prob.sigmoid().cpu().numpy().squeeze()
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
            pred_bin = (prob >= 0.5).astype(np.uint8)
            binary = pred_bin * 255
            image_rgb = _load_rgb(image_path, target_shape=gt_bin.shape)
            contour = draw_prediction_gt_contours(image_rgb, pred_bin, gt_bin)

            cv2.imwrite(os.path.join(prob_dir, name), (prob * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(binary_dir, name), binary)
            cv2.imwrite(
                os.path.join(contour_dir, name),
                cv2.cvtColor(contour, cv2.COLOR_RGB2BGR),
            )

        saved[dataset] = {
            "prob": prob_dir,
            "binary": binary_dir,
            "contours": contour_dir,
            "count": loader.size,
        }

    return saved
