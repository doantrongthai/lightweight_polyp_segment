"""
main.py
────────────────────────────────────────────────────────────────
Changes vs previous version:
  1. --loss default uses 'structure' (weighted BCE + weighted IoU),
     matching the binary Polyp-PVT/PraNet training objective.
  2. Tversky/Focal variants are kept for ablation.
"""

import os
import sys
import argparse
import json
import torch

from utils.seed import seed_everything, log_versions
from utils.utils import cal_params
from utils.dataset import prepare_datasets
from losses import list_losses

LOSS_DEFAULT = 'structure'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str, required=True)
    parser.add_argument('--train_path', type=str, default='./dataset')
    parser.add_argument('--test_path',  type=str, default='./dataset')
    parser.add_argument('--test', dest='test_dataset', type=str, default='all',
                        choices=['CVC-300', 'CVC-ClinicDB', 'Kvasir',
                                 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'all'])

    # ── Paper Table II hyperparameters ────────────────────────────────
    parser.add_argument('--epoch',       type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--batchsize',   type=int,   default=16)
    parser.add_argument('--trainsize',   type=int,   default=352,
                        help='fixed training/input size; forced to 352')
    parser.add_argument('--clip',        type=float, default=0.5)
    parser.add_argument('--decay_rate',  type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int,   default=50)
    parser.add_argument('--optimizer',   type=str,   default='AdamW',
                        choices=['AdamW', 'SGD'])

    # ── Loss ─────────────────────────────────────────────────────────
    # Auto-detected from losses/*.py via @register_loss (see losses/__init__.py),
    # same mechanism as --model. Add a new loss file with @register_loss("name")
    # and it becomes available here automatically — no edits needed in main.py.
    parser.add_argument('--loss',        type=str,   default=LOSS_DEFAULT,
                        help=f"Loss to train with (default: {LOSS_DEFAULT}). "
                             f"Available (auto-detected): {list_losses()}")

    parser.add_argument('--augmentation', type=lambda x: x != 'False', default=True)
    parser.add_argument('--eval_freq',   type=int,   default=1)
    parser.add_argument('--save_path',   type=str,   default='./checkpoints')
    parser.add_argument('--val_ratio',   type=float, default=0.2,
                        help='validation split ratio from TrainDataset')
    parser.add_argument('--split_seed',  type=int,   default=42,
                        help='fixed seed for train/val split')

    # ── Seed / multi-seed ────────────────────────────────────────────
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--multiseed',   action='store_true')
    parser.add_argument('--seeds',       type=int,   nargs='+',
                        default=[42, 123, 456, 789, 2024])

    # ── Resume & inference ───────────────────────────────────────────
    parser.add_argument('--resume',      action='store_true')
    parser.add_argument('--test_only',   action='store_true')
    parser.add_argument('--weights',     type=str,   default=None)
    parser.add_argument('--visualize',   action='store_true',
                        help='with --test_only: save prediction maps and Grad-CAM panels')
    parser.add_argument('--save_maps',   action='store_true',
                        help='with --test_only: save probability and binary prediction maps')
    parser.add_argument('--gradcam',     action='store_true',
                        help='with --test_only: save Grad-CAM best/worst panels')
    parser.add_argument('--vis_path',    type=str,   default='./visualizations')
    parser.add_argument('--gradcam_k',   type=int,   default=10)

    # ── T4 GPU optimisation ───────────────────────────────────────────
    parser.add_argument('--amp',         action='store_true')
    parser.add_argument('--num_workers', type=int,
                        default=min(4, (os.cpu_count() or 1)))

    # ── Dataset ──────────────────────────────────────────────────────
    parser.add_argument('--skip_download', action='store_true')

    return parser.parse_args()


def _run_test_visualizations(opt, model, device, all_datasets):
    do_maps = opt.visualize or opt.save_maps
    do_gradcam = opt.visualize or opt.gradcam
    if not do_maps and not do_gradcam:
        return

    datasets = all_datasets if opt.test_dataset == 'all' else [opt.test_dataset]
    root = os.path.join(opt.vis_path, opt.model, f'seed{opt.seed}')
    os.makedirs(root, exist_ok=True)

    if do_maps:
        from utils.visualize import save_prediction_maps
        maps_root = os.path.join(root, 'prediction_maps')
        saved = save_prediction_maps(
            model=model,
            test_path=opt.test_path,
            datasets=datasets,
            testsize=opt.trainsize,
            device=device,
            output_root=maps_root,
        )
        for dataset, info in saved.items():
            print(f'[visualize] {dataset}: saved {info["count"]} maps to {maps_root}')

    if do_gradcam:
        from utils.gradcam import run_analysis
        for dataset in datasets:
            gradcam_opt = argparse.Namespace(
                model=opt.model,
                weights=opt.weights,
                test_path=opt.test_path,
                dataset=dataset,
                testsize=opt.trainsize,
                k=opt.gradcam_k,
                output_dir=os.path.join(root, 'gradcam', dataset),
            )
            run_analysis(gradcam_opt, model=model, device=device)


def main():
    opt = parse_args()
    if not (0.0 < opt.val_ratio < 1.0):
        raise ValueError('--val_ratio must be between 0 and 1')
    if opt.loss.lower() not in list_losses():
        raise ValueError(
            f"--loss '{opt.loss}' not found. Available: {list_losses()}"
        )
    opt.loss_is_default = (opt.loss == LOSS_DEFAULT)
    opt.trainsize = 352
    opt.save_path = os.path.join(opt.save_path, opt.model)
    os.makedirs(opt.save_path, exist_ok=True)

    max_workers = os.cpu_count() or 1
    if opt.num_workers > max_workers:
        print(f'[warn] num_workers capped {opt.num_workers}→{max_workers}')
        opt.num_workers = max_workers

    log_versions()

    if not opt.skip_download:
        dataset_root = opt.train_path  # FIX: dùng thẳng, không dùng dirname
        prepare_datasets(
            root       = dataset_root,
            need_train = not opt.test_only,
            need_test  = True,
        )

    seed_everything(opt.seed)

    from models import get_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print(f'Model      : {opt.model}')
    print(f'Test set   : {opt.test_dataset}')
    print(f'Epochs     : {opt.epoch} | BS: {opt.batchsize} | LR: {opt.lr}')
    default_tag = '(default)' if opt.loss_is_default else '(custom)'
    print(f'Loss       : {opt.loss} {default_tag} | Optimizer: {opt.optimizer}')
    print(f'Scheduler  : ReduceLROnPlateau (patience=10, factor=0.5)')
    print(f'Train size : {opt.trainsize}x{opt.trainsize} | Device: {device}')
    print(f'Val split  : {1 - opt.val_ratio:.0%}/{opt.val_ratio:.0%} '
          f'from train | split_seed={opt.split_seed}')
    print(f'AMP (FP16) : {opt.amp}')
    print(f'Resume     : {opt.resume}')
    print(f'Visualize  : {opt.visualize} | Maps: {opt.save_maps} | GradCAM: {opt.gradcam}')
    print(f'num_workers: {opt.num_workers}')
    print(f'Save path  : {opt.save_path}')
    print('=' * 70)

    if opt.test_only:
        assert opt.weights is not None, '--weights required with --test_only'
        from utils.trainer import TEST_DATASETS, test
        model = get_model(opt.model).to(device)
        test(opt, model, opt.weights, opt.seed)
        _run_test_visualizations(opt, model, device, TEST_DATASETS)
        return

    if opt.multiseed:
        from utils.trainer import multiseed_train_test
        def model_fn():
            return get_model(opt.model)
        multiseed_train_test(opt, model_fn, seeds=tuple(opt.seeds))
        return

    from utils.trainer import train, test

    seed_everything(opt.seed)
    model = get_model(opt.model).to(device)

    flops, params = cal_params(model, input_size=(1, 3, opt.trainsize, opt.trainsize))
    print(f'FLOPs: {flops} | Params: {params}')

    best_path, history = train(opt, model, opt.seed)

    history_path = os.path.join(opt.save_path, f'history_seed{opt.seed}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[main] History saved: {history_path}')

    try:
        from plot import plot_history
        plot_history(history, opt.save_path, opt.model, opt.seed)
    except Exception as e:
        print(f'[main] plot_history skipped: {e}')

    seed_everything(opt.seed)
    test(opt, model, best_path, opt.seed)
    print(f'\n[main] Best weights: {best_path}')


if __name__ == '__main__':
    main()
