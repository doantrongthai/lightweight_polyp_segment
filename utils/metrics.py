import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import roc_auc_score


def _to_binary(pred, gt, threshold=0.5):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred = (pred >= threshold).astype(np.float32)
    gt = (gt >= 0.5).astype(np.float32)
    return pred.flatten(), gt.flatten()


def dice_score(pred, gt, threshold=0.5, smooth=1e-8):
    p, g = _to_binary(pred, gt, threshold)
    intersection = (p * g).sum()
    return (2.0 * intersection + smooth) / (p.sum() + g.sum() + smooth)


def iou_score(pred, gt, threshold=0.5, smooth=1e-8):
    p, g = _to_binary(pred, gt, threshold)
    intersection = (p * g).sum()
    union = p.sum() + g.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(pred, gt, threshold=0.5, smooth=1e-8):
    p, g = _to_binary(pred, gt, threshold)
    tp = (p * g).sum()
    return (tp + smooth) / (p.sum() + smooth)


def recall_score(pred, gt, threshold=0.5, smooth=1e-8):
    p, g = _to_binary(pred, gt, threshold)
    tp = (p * g).sum()
    return (tp + smooth) / (g.sum() + smooth)


def specificity_score(pred, gt, threshold=0.5, smooth=1e-8):
    p, g = _to_binary(pred, gt, threshold)
    tn = ((1 - p) * (1 - g)).sum()
    fp = (p * (1 - g)).sum()
    return (tn + smooth) / (tn + fp + smooth)


def balanced_accuracy(pred, gt, threshold=0.5, smooth=1e-8):
    rec = recall_score(pred, gt, threshold, smooth)
    spec = specificity_score(pred, gt, threshold, smooth)
    return (rec + spec) / 2.0


def mcc_score(pred, gt, threshold=0.5, smooth=1e-8):
    p, g = _to_binary(pred, gt, threshold)
    tp = (p * g).sum()
    tn = ((1 - p) * (1 - g)).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + smooth
    return float(num / den)


def auc_roc_score(pred, gt):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred_flat = pred.flatten().astype(np.float64)
    gt_flat = (gt.flatten() >= 0.5).astype(np.int32)
    if gt_flat.sum() == 0 or gt_flat.sum() == len(gt_flat):
        return 0.5
    return float(roc_auc_score(gt_flat, pred_flat))


def mae_score(pred, gt):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred = pred.astype(np.float32)
    gt = (gt >= 0.5).astype(np.float32)
    return np.mean(np.abs(pred - gt))


def weighted_fmeasure(pred, gt, beta2=1.0, smooth=1e-8):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred = pred.astype(np.float64)
    gt = (gt >= 0.5).astype(np.float64)

    if gt.max() == 0:
        return 0.0

    q = distance_transform_edt(gt) + distance_transform_edt(1 - gt)
    q = q / (q.max() + smooth)
    tpr_w = (pred * gt * q).sum() / (gt * q).sum() if (gt * q).sum() > 0 else 0.0
    fpr_w = (pred * (1 - gt) * q).sum() / ((1 - gt) * q).sum() if ((1 - gt) * q).sum() > 0 else 0.0
    prec_w = tpr_w / (tpr_w + fpr_w + smooth)
    rec_w = tpr_w
    fw = (1 + beta2) * prec_w * rec_w / (beta2 * prec_w + rec_w + smooth)
    return float(fw)


def s_measure(pred, gt, alpha=0.5):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred = pred.astype(np.float64)
    gt = (gt >= 0.5).astype(np.float64)

    y = gt.mean()
    if y == 0:
        return 1.0 - pred.mean()
    if y == 1:
        return pred.mean()

    so = _s_object(pred, gt)
    sr = _s_region(pred, gt)
    return alpha * so + (1 - alpha) * sr


def _s_object(pred, gt):
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)
    u = gt.mean()
    op = _object(fg, gt)
    oq = _object(bg, 1 - gt)
    return u * op + (1 - u) * oq


def _object(pred, gt):
    smooth = 1e-8
    tp = (pred * gt).sum()
    prec = tp / (pred.sum() + smooth)
    rec = tp / (gt.sum() + smooth)
    return 2.0 * prec * rec / (prec + rec + smooth)


def _s_region(pred, gt):
    x = (gt * np.arange(gt.shape[1])).sum() / (gt.sum() + 1e-8)
    y = (gt * np.arange(gt.shape[0])[:, None]).sum() / (gt.sum() + 1e-8)
    x, y = int(round(x)), int(round(y))

    w1 = gt[:y, :x].mean() if gt[:y, :x].size > 0 else 0
    w2 = gt[:y, x:].mean() if gt[:y, x:].size > 0 else 0
    w3 = gt[y:, :x].mean() if gt[y:, :x].size > 0 else 0
    w4 = gt[y:, x:].mean() if gt[y:, x:].size > 0 else 0

    q1 = _ssim(pred[:y, :x], gt[:y, :x]) if gt[:y, :x].size > 0 else 0
    q2 = _ssim(pred[:y, x:], gt[:y, x:]) if gt[:y, x:].size > 0 else 0
    q3 = _ssim(pred[y:, :x], gt[y:, :x]) if gt[y:, :x].size > 0 else 0
    q4 = _ssim(pred[y:, x:], gt[y:, x:]) if gt[y:, x:].size > 0 else 0

    return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4


def _ssim(pred, gt):
    smooth = 1e-8
    n = pred.size
    if n == 0:
        return 0.0
    mu_p, mu_g = pred.mean(), gt.mean()
    sig_p = np.sqrt(((pred - mu_p) ** 2).mean())
    sig_g = np.sqrt(((gt - mu_g) ** 2).mean())
    sig_pg = ((pred - mu_p) * (gt - mu_g)).mean()
    return (2 * mu_p * mu_g + smooth) * (2 * sig_pg + smooth) / \
           ((mu_p**2 + mu_g**2 + smooth) * (sig_p**2 + sig_g**2 + smooth))


def e_measure(pred, gt, threshold=0.5):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred_bin = (pred >= threshold).astype(np.float64)
    gt = (gt >= 0.5).astype(np.float64)
    mu_p = pred_bin.mean()
    mu_g = gt.mean()
    align = 2 * (pred_bin - mu_p) * (gt - mu_g) / \
            (((pred_bin - mu_p)**2 + (gt - mu_g)**2) + 1e-8)
    enhanced = (align + 1)**2 / 4
    return enhanced.mean()


def hd95_score(pred, gt, threshold=0.5):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin = (gt >= 0.5).astype(np.uint8)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float(max(pred_bin.shape))

    dist_pred = distance_transform_edt(1 - gt_bin)
    dist_gt = distance_transform_edt(1 - pred_bin)

    surface_pred = dist_pred[pred_bin == 1]
    surface_gt = dist_gt[gt_bin == 1]

    if len(surface_pred) == 0 or len(surface_gt) == 0:
        pred_pts = np.argwhere(pred_bin)
        gt_pts = np.argwhere(gt_bin)
        d1 = directed_hausdorff(pred_pts, gt_pts)[0]
        d2 = directed_hausdorff(gt_pts, pred_pts)[0]
        return float(max(d1, d2))

    return float(np.percentile(np.concatenate([surface_pred, surface_gt]), 95))


def nsd_score(pred, gt, threshold=0.5, tau=2.0):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin = (gt >= 0.5).astype(np.uint8)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0

    pred_boundary = _extract_boundary(pred_bin)
    gt_boundary = _extract_boundary(gt_bin)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    dist_gt = distance_transform_edt(1 - gt_bin)
    dist_pred = distance_transform_edt(1 - pred_bin)

    pred_within = (dist_gt[pred_boundary == 1] <= tau).sum()
    gt_within = (dist_pred[gt_boundary == 1] <= tau).sum()

    num = pred_within + gt_within
    den = pred_boundary.sum() + gt_boundary.sum()
    return float(num / (den + 1e-8))


def _extract_boundary(binary_mask):
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(binary_mask, iterations=1)
    return (binary_mask.astype(np.int32) - eroded.astype(np.int32)).clip(0, 1).astype(np.uint8)


def compute_all_metrics(pred_np, gt_np, threshold=0.5, nsd_tau=2.0):
    pred = pred_np.astype(np.float64)
    gt = gt_np.astype(np.float64)
    return {
        'dice':             dice_score(pred, gt, threshold),
        'iou':              iou_score(pred, gt, threshold),
        'precision':        precision_score(pred, gt, threshold),
        'recall':           recall_score(pred, gt, threshold),
        'specificity':      specificity_score(pred, gt, threshold),
        'balanced_acc':     balanced_accuracy(pred, gt, threshold),
        'mcc':              mcc_score(pred, gt, threshold),
        'auc_roc':          auc_roc_score(pred, gt),
        'mae':              mae_score(pred, gt),
        'fw':               weighted_fmeasure(pred, gt),
        'sm':               s_measure(pred, gt),
        'em':               e_measure(pred, gt, threshold),
        'hd95':             hd95_score(pred, gt, threshold),
        'nsd':              nsd_score(pred, gt, threshold, nsd_tau),
    }


def aggregate_metrics(metric_list):
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}
