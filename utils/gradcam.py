import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.dataloader import TestDataset
from utils.metrics import dice_score, iou_score
from utils.visualize import draw_prediction_gt_contours
from utils.utils import split_model_output, fuse_outputs


class SegGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._hooks.append(target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()

        out = self.model(input_tensor)
        main_out, aux_outputs = split_model_output(out)
        main = fuse_outputs(main_out, aux_outputs)

        pred_prob = torch.sigmoid(main)
        pred_mask = (pred_prob > 0.5).float()
        score = (main * pred_mask).sum()

        if score.item() == 0:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w)), pred_prob.detach().cpu().numpy()[0, 0]

        score.backward()

        grads = self.gradients.cpu().numpy()[0]
        acts = self.activations.cpu().numpy()[0]
        weights = grads.mean(axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_prob.detach().cpu().numpy()[0, 0]


def _get_target_layer(model):
    candidates = ['d1', 'decoder', 'head', 'out_CFM', 'seg_head']
    for name in candidates:
        if hasattr(model, name):
            return getattr(model, name)
    # FIX: tìm last conv layer thực sự
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is not None:
        return last
    # fallback cuối
    return list(model.children())[-1]


def _error_map(pred_bin, gt_bin):
    rgb = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
    tp = (pred_bin == 1) & (gt_bin == 1)
    fp = (pred_bin == 1) & (gt_bin == 0)
    fn = (pred_bin == 0) & (gt_bin == 1)
    rgb[tp] = [0, 200, 0]
    rgb[fp] = [220, 0, 0]
    rgb[fn] = [0, 0, 220]
    return rgb


def _denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy()[0].transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _save_panel(item, rank, prefix, grad_cam, device, output_dir, testsize):
    img_tensor = item['img_tensor'].to(device)
    gt_np = item['gt']
    iou = item['iou']
    dice = item['dice']

    cam, pred_prob = grad_cam.generate(img_tensor)
    pred_bin = (pred_prob > 0.5).astype(np.uint8)
    gt_bin = (gt_np >= 0.5).astype(np.uint8)

    img_show = _denorm(item['img_tensor'])
    img_show_resized = cv2.resize(img_show, (testsize, testsize))

    cam_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_show_resized, 0.5, cam_color, 0.5, 0)

    pred_bin_resized = cv2.resize(pred_bin.astype(np.uint8),
                                   (testsize, testsize), interpolation=cv2.INTER_NEAREST)
    gt_resized = cv2.resize(gt_bin.astype(np.uint8),
                             (testsize, testsize), interpolation=cv2.INTER_NEAREST)
    pred_only = draw_prediction_gt_contours(
        img_show_resized, pred_bin_resized, np.zeros_like(gt_resized)
    )
    gt_only = draw_prediction_gt_contours(
        img_show_resized, np.zeros_like(pred_bin_resized), gt_resized
    )
    contour_overlay = draw_prediction_gt_contours(
        img_show_resized, pred_bin_resized, gt_resized
    )
    gradcam_contours = draw_prediction_gt_contours(
        overlay, pred_bin_resized, gt_resized
    )

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = [
        'Image',
        'Prediction (green contour)',
        'Ground Truth (red contour)',
        'Pred + GT contours',
        'Grad-CAM + contours',
    ]
    data = [img_show_resized, pred_only, gt_only, contour_overlay, gradcam_contours]
    cmaps = [None, None, None, None, None]

    for ax, title, d, cmap in zip(axes, titles, data, cmaps):
        ax.imshow(d, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig.suptitle(f'{prefix} #{rank+1} | IoU={iou:.4f} | Dice={dice:.4f}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{prefix.lower()}_{rank+1:02d}_iou{iou:.3f}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()


def _plot_iou_distribution(results, output_dir):
    ious = [r['iou'] for r in results]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(ious, bins=40, edgecolor='black', color='steelblue', alpha=0.8)
    ax.axvline(np.mean(ious), color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('IoU')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_distribution.png'), dpi=120, bbox_inches='tight')
    plt.close()


def run_analysis(opt, model=None, device=None):
    os.makedirs(opt.output_dir, exist_ok=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = os.path.join(opt.test_path, opt.dataset)
    image_root = os.path.join(data_path, 'images', '')
    gt_root = os.path.join(data_path, 'masks', '')
    loader = TestDataset(image_root, gt_root, opt.testsize)

    if model is None:
        from models import get_model
        model = get_model(opt.model).to(device)
        from utils.checkpoint import load_model_weights
        ckpt = torch.load(opt.weights, map_location=device, weights_only=False)
        load_model_weights(model, ckpt)
    else:
        model = model.to(device)
    model.eval()

    target_layer = _get_target_layer(model)
    grad_cam = SegGradCAM(model, target_layer)

    results = []
    print(f'Scoring {loader.size} images...')
    for idx in tqdm(range(loader.size)):
        img_tensor, gt_pil, name = loader.load_data()
        gt_np = np.asarray(gt_pil, np.float32)
        gt_np = gt_np / (gt_np.max() + 1e-8)
        img = img_tensor.to(device)

        with torch.no_grad():
            out = model(img)
            main_out, aux_outputs = split_model_output(out)
            main = fuse_outputs(main_out, aux_outputs)
            res = F.interpolate(main, size=gt_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        results.append({
            'id':         idx,
            'name':       name,
            'img_tensor': img_tensor.cpu(),
            'gt':         gt_np,
            'iou':        iou_score(res, gt_np),
            'dice':       dice_score(res, gt_np),
        })

    results.sort(key=lambda x: x['iou'])
    worst = results[:opt.k]
    best = results[-opt.k:][::-1]

    print('Generating BEST cases...')
    for rank, item in enumerate(best):
        _save_panel(item, rank, 'BEST', grad_cam, device, opt.output_dir, opt.testsize)

    print('Generating WORST cases...')
    for rank, item in enumerate(worst):
        _save_panel(item, rank, 'WORST', grad_cam, device, opt.output_dir, opt.testsize)

    _plot_iou_distribution(results, opt.output_dir)
    grad_cam.remove_hooks()

    all_iou = [r['iou'] for r in results]
    all_dice = [r['dice'] for r in results]
    print(f'\nDataset: {opt.dataset} | N={len(results)}')
    print(f'IoU:  mean={np.mean(all_iou):.4f} std={np.std(all_iou):.4f}')
    print(f'Dice: mean={np.mean(all_dice):.4f} std={np.std(all_dice):.4f}')
    print(f'Saved to: {opt.output_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--test_path', type=str, default='./dataset/TestDataset')
    parser.add_argument('--dataset', type=str, default='Kvasir',
                        choices=['CVC-300', 'CVC-ClinicDB', 'Kvasir',
                                 'CVC-ColonDB', 'ETIS-LaribPolypDB'])
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./gradcam_results')
    opt = parser.parse_args()
    opt.testsize = 352
    run_analysis(opt)


if __name__ == '__main__':
    main()
