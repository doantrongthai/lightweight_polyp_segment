"""
utils/trainer.py
────────────────────────────────────────────────────────────────
Changes vs previous version:
  1. [SCHEDULER]  CosineAnnealingLR → ReduceLROnPlateau.
     - mode='max', monitors val Dice.
     - factor=0.5 (halve LR on plateau), patience=10 epochs,
       min_lr=1e-6.  Matches FCBFormer (2022) and is now the
       dominant choice in 2024-2025 polyp papers.
     - scheduler.step() is called ONLY on eval epochs with the
       actual val Dice value — NOT every epoch unconditionally.
       Calling ReduceLROnPlateau every epoch when the metric
       hasn't been refreshed would count non-eval epochs as
       "no improvement" and trigger premature LR decay.
  2. [LOSS]       Default loss is 'structure' (weighted BCE + weighted IoU).
  3. [RESUME]     Scheduler state saved/loaded correctly for
                  ReduceLROnPlateau (state dict format differs
                  from CosineAnnealingLR but torch handles it).
  4. [MAIN.PY]    --loss default changed to 'structure'
                  (see main.py); choices list updated.
"""

import os
import gc
import json
import logging
import importlib
import math
import platform
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from datetime import datetime

from utils.dataloader import get_train_val_loaders, TestDataset
from utils.metrics import compute_all_metrics, aggregate_metrics
from utils.efficiency import full_efficiency_report
from utils.stats import multiseed_stats, latex_table, print_multiseed_report
from utils.utils import (
    clip_gradient, AvgMeter, cal_params,
    split_model_output, fuse_outputs, compute_loss_with_aux,
)
from utils.seed import seed_everything, seed_epoch
from utils.checkpoint import load_model_weights, model_state_dict


TEST_DATASETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
FIXED_IMAGE_SIZE = 352


# ─────────────────────────────────────────────────────────────────────────────
# Forward helpers
# ─────────────────────────────────────────────────────────────────────────────

def _forward(model, images):
    return split_model_output(model(images))


def _fuse(main, aux):
    """
    Final prediction used for eval/inference metrics.
    See utils.utils.fuse_outputs for the exact rule per architecture
    (P1+P2 style dual-branch fusion vs. PraNet-style deep supervision).
    """
    return fuse_outputs(main, aux)


def _loss_with_aux(criterion, main, aux, masks):
    """L = L_main + sum(L_aux_i), equal weights (see utils.utils.compute_loss_with_aux).
    For PraNet's 4-way deep supervision this reproduces the original
    loss2 + loss3 + loss4 + loss5 exactly."""
    return compute_loss_with_aux(criterion, main, aux, masks)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_dice, history, scaler=None):
    payload = {
        'epoch':           epoch,
        'best_dice':       best_dice,
        'history':         history,
        'model_state':     model_state_dict(model),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    if scaler is not None:
        payload['scaler_state'] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, scheduler, device, scaler=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    load_model_weights(model, ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    if scaler is not None and 'scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state'])
    return ckpt['epoch'], ckpt['best_dice'], ckpt['history']


def _package_version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, '__version__', 'unknown')
    except Exception:
        return None


def _git_output(args):
    try:
        result = subprocess.run(
            ['git'] + args,
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _git_info():
    status = _git_output(['status', '--porcelain'])
    return {
        'commit': _git_output(['rev-parse', 'HEAD']),
        'branch': _git_output(['rev-parse', '--abbrev-ref', 'HEAD']),
        'dirty': bool(status),
        'status': status.splitlines() if status else [],
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _save_run_config(opt, model, optimizer, scheduler, seed, split_info,
                     split_path, best_path, resume_ckpt, device):
    cuda_available = torch.cuda.is_available()
    config = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'command': {
            'executable': sys.executable,
            'argv': list(sys.argv),
            'cwd': os.getcwd(),
        },
        'args': dict(vars(opt)),
        'seed': int(seed),
        'split': split_info,
        'artifacts': {
            'save_path': opt.save_path,
            'split_file': split_path,
            'best_checkpoint': best_path,
            'resume_checkpoint': resume_ckpt,
        },
        'model': {
            'name': getattr(opt, 'model', model.__class__.__name__),
            'class': model.__class__.__name__,
            'total_params': int(sum(p.numel() for p in model.parameters())),
            'trainable_params': int(sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)),
        },
        'training': {
            'optimizer': optimizer.__class__.__name__,
            'optimizer_state': optimizer.defaults,
            'scheduler': scheduler.__class__.__name__,
            'scheduler_state': scheduler.state_dict(),
            'loss': getattr(opt, 'loss', None),
            'amp': bool(getattr(opt, 'amp', False)),
            'trainsize': int(getattr(opt, 'trainsize', FIXED_IMAGE_SIZE)),
            'fixed_image_size': FIXED_IMAGE_SIZE,
        },
        'runtime': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'device': str(device),
            'cuda_available': cuda_available,
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count() if cuda_available else 0,
            'gpu': torch.cuda.get_device_name(0) if cuda_available else None,
            'packages': {
                name: _package_version(name)
                for name in ('torch', 'torchvision', 'timm', 'numpy',
                             'sklearn', 'scipy', 'thop')
            },
        },
        'reproducibility': {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
            'mkl_num_threads': os.environ.get('MKL_NUM_THREADS'),
        },
        'git': _git_info(),
    }

    path = os.path.join(opt.save_path, f'run_config_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(_json_safe(config), f, indent=2, allow_nan=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Validation (Dice / IoU / MAE only — fast)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_dataset(model, test_path, dataset, testsize, device):
    data_path  = os.path.join(test_path, dataset)
    image_root = os.path.join(data_path, 'images', '')
    gt_root    = os.path.join(data_path, 'masks',  '')
    loader = TestDataset(image_root, gt_root, testsize)
    model.eval()

    from utils.metrics import dice_score, iou_score, mae_score
    dice_vals, iou_vals, mae_vals = [], [], []

    for _ in range(loader.size):
        image, gt, _ = loader.load_data()
        gt_np = np.asarray(gt, np.float32)
        gt_np = gt_np / (gt_np.max() + 1e-8)
        image = image.to(device)

        main_out, aux_outputs = _forward(model, image)
        fused = _fuse(main_out, aux_outputs)
        res = F.interpolate(fused, size=gt_np.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        dice_vals.append(dice_score(res, gt_np))
        iou_vals.append(iou_score(res, gt_np))
        mae_vals.append(mae_score(res, gt_np))

    return {
        'dice': float(np.mean(dice_vals)),
        'iou':  float(np.mean(iou_vals)),
        'mae':  float(np.mean(mae_vals)),
    }


@torch.no_grad()
def eval_val_loader(model, val_loader, device):
    model.eval()
    from utils.metrics import dice_score, iou_score, mae_score
    dice_vals, iou_vals, mae_vals = [], [], []

    for images_cpu, gts_cpu in val_loader:
        images = images_cpu.to(device)
        main_out, aux_outputs = _forward(model, images)
        fused = _fuse(main_out, aux_outputs)
        if fused.shape[2:] != gts_cpu.shape[2:]:
            fused = F.interpolate(
                fused, size=gts_cpu.shape[2:], mode='bilinear',
                align_corners=False)
        probs = fused.sigmoid().cpu().numpy()
        gts = gts_cpu.numpy()

        for idx in range(probs.shape[0]):
            pred = probs[idx].squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            gt = gts[idx].squeeze()
            dice_vals.append(dice_score(pred, gt))
            iou_vals.append(iou_score(pred, gt))
            mae_vals.append(mae_score(pred, gt))

    return {
        'dice': float(np.mean(dice_vals)),
        'iou':  float(np.mean(iou_vals)),
        'mae':  float(np.mean(mae_vals)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer, criterion,
                    opt, epoch, total_step, logger, device, scaler=None):
    model.train()
    size_rates  = [1.0]
    loss_record = AvgMeter()

    for i, (images_cpu, gts_cpu) in enumerate(train_loader, start=1):
        images_orig = images_cpu.to(device)
        gts_orig    = gts_cpu.to(device)

        for rate in size_rates:
            optimizer.zero_grad()

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images_orig,
                                       size=(trainsize, trainsize),
                                       mode='bilinear', align_corners=True)
                gts    = F.interpolate(gts_orig,
                                       size=(trainsize, trainsize),
                                       mode='bilinear', align_corners=True)
            else:
                images, gts = images_orig, gts_orig

            if scaler is not None:
                with autocast():
                    main_out, aux_outputs = _forward(model, images)
                    loss = _loss_with_aux(criterion, main_out, aux_outputs, gts)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, opt.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                main_out, aux_outputs = _forward(model, images)
                loss = _loss_with_aux(criterion, main_out, aux_outputs, gts)
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            msg = (f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch}] '
                   f'Step [{i:04d}/{total_step}] Loss: {loss_record.show():.4f}')
            print(msg)
            logger.info(msg)

    return loss_record.avg


def train(opt, model, seed):
    seed_everything(seed)
    opt.trainsize = FIXED_IMAGE_SIZE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_amp = getattr(opt, 'amp', False) and device.type == 'cuda'
    scaler  = GradScaler() if use_amp else None
    if use_amp:
        print('[AMP] FP16 enabled — GradScaler active')

    os.makedirs(opt.save_path, exist_ok=True)
    log_file = os.path.join(opt.save_path, f'train_seed{seed}.log')
    logging.basicConfig(filename=log_file,
                        format='[%(asctime)s %(filename)s %(levelname)s] %(message)s',
                        level=logging.INFO, filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')
    logger = logging.getLogger(__name__ + str(seed))

    # FIX: 'image' không có 's' — khớp với cấu trúc thực tế của dataset
    image_root   = os.path.join(opt.train_path, 'image', '')
    gt_root      = os.path.join(opt.train_path, 'masks', '')
    train_loader, val_loader, split_info = get_train_val_loaders(
        image_root, gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        val_ratio=getattr(opt, 'val_ratio', 0.2),
        split_seed=getattr(opt, 'split_seed', 42),
        augmentation=opt.augmentation,
        seed=seed,
        num_workers=getattr(opt, 'num_workers', None),
    )
    total_step = len(train_loader)

    split_path = os.path.join(
        opt.save_path,
        f'val_split_seed{split_info["split_seed"]}_ratio{split_info["val_ratio"]:.2f}.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f'[split] Train={split_info["train_size"]} | '
          f'Val={split_info["val_size"]} | '
          f'Seed={split_info["split_seed"]} | saved: {split_path}')

    from losses import get_loss
    criterion = get_loss(opt.loss)

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr,
                                    weight_decay=1e-4, momentum=0.9)

    # ── CHANGE: ReduceLROnPlateau replaces CosineAnnealingLR ──────────
    # mode='max': we want Dice to go UP (not down like loss).
    # factor=0.5: halve the LR each time.
    # patience=10: wait 10 eval epochs with no improvement before decay.
    #   With eval_freq=1 this equals 10 epochs; adjust if eval_freq > 1.
    # min_lr=1e-6: floor to avoid LR becoming uselessly small.
    # verbose=True: prints "[Epoch N] Reducing LR to X" in console.
    #
    # IMPORTANT: scheduler.step(val_dice) is called ONLY inside the
    # eval block, NOT unconditionally every epoch.  Calling it every
    # epoch without a fresh metric would count non-eval epochs as
    # "no improvement" and trigger premature LR decay.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    resume_ckpt = os.path.join(opt.save_path, f'resume_seed{seed}.pth')
    best_path   = os.path.join(opt.save_path, f'best_seed{seed}.pth')
    config_path = _save_run_config(
        opt=opt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        seed=seed,
        split_info=split_info,
        split_path=split_path,
        best_path=best_path,
        resume_ckpt=resume_ckpt,
        device=device,
    )
    print(f'[config] Run config saved: {config_path}')
    logger.info(f'Run config saved: {config_path}')

    start_epoch = 1
    best_dice   = -1.0
    history     = {'val': []}

    if getattr(opt, 'resume', False) and os.path.exists(resume_ckpt):
        print(f'\n[resume] Found checkpoint: {resume_ckpt}')
        start_epoch, best_dice, history = load_checkpoint(
            resume_ckpt, model, optimizer, scheduler, device, scaler)
        history.setdefault('val', [])
        start_epoch += 1
        print(f'[resume] Resuming from epoch {start_epoch}/{opt.epoch} '
              f'| best_dice={best_dice:.4f} | AMP={use_amp}\n')
        logger.info(f'Resumed from epoch {start_epoch - 1}, best_dice={best_dice:.4f}')
    elif getattr(opt, 'resume', False):
        print(f'[resume] No checkpoint found at {resume_ckpt} — training from scratch.\n')

    if start_epoch > opt.epoch:
        print('[train] Already trained for target number of epochs.')
        return best_path, history

    # Print current LR at start
    cur_lr = optimizer.param_groups[0]['lr']
    print(f'[train] Loss: {opt.loss} | Scheduler: ReduceLROnPlateau '
          f'(patience={scheduler.patience}, factor={scheduler.factor}) '
          f'| LR: {cur_lr:.2e}')

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, opt.epoch + 1):
        seed_epoch(seed, epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     opt, epoch, total_step, logger, device, scaler)

        # ── Eval + scheduler step ─────────────────────────────────────
        # scheduler.step() is ONLY called here, with the actual val Dice.
        if epoch % opt.eval_freq == 0:
            m_val = eval_val_loader(model, val_loader, device)
            history['val'].append(m_val['dice'])

            # ── Step scheduler with val Dice ──────────────────────────
            scheduler.step(m_val['dice'])
            cur_lr = optimizer.param_groups[0]['lr']

            header = (f'Epoch [{epoch:03d}/{opt.epoch}]  '
                      f'TrainLoss={train_loss:.4f}  '
                      f'Val(split) Dice={m_val["dice"]:.4f}  '
                      f'IoU={m_val["iou"]:.4f}  MAE={m_val["mae"]:.4f}  '
                      f'LR={cur_lr:.2e}')
            print(f'\n{"-"*70}')
            print(header)
            print(f'{"-"*70}')
            logger.info(header)

            if m_val['dice'] > best_dice:
                best_dice = m_val['dice']
                torch.save(model_state_dict(model), best_path)
                msg = f'  >>> Best updated on val split: Dice={best_dice:.4f} (seed={seed})'
                print(msg)
                logger.info(msg)

        # Save resume checkpoint every epoch (includes scheduler state)
        save_checkpoint(resume_ckpt, model, optimizer, scheduler,
                        epoch, best_dice, history, scaler)

    if os.path.exists(resume_ckpt):
        os.remove(resume_ckpt)
        print('\n[train] Training complete. Resume checkpoint removed.')

    return best_path, history


# ─────────────────────────────────────────────────────────────────────────────
# Test (full metrics)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_full(model, test_path, dataset, testsize, device):
    data_path  = os.path.join(test_path, dataset)
    image_root = os.path.join(data_path, 'images', '')
    gt_root    = os.path.join(data_path, 'masks',  '')
    loader = TestDataset(image_root, gt_root, testsize)
    model.eval()
    metrics_list = []

    for _ in range(loader.size):
        image, gt, _ = loader.load_data()
        gt_np = np.asarray(gt, np.float32)
        gt_np = gt_np / (gt_np.max() + 1e-8)
        image = image.to(device)

        main_out, aux_outputs = _forward(model, image)
        fused = _fuse(main_out, aux_outputs)
        res = F.interpolate(fused, size=gt_np.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        metrics_list.append(compute_all_metrics(res, gt_np))

    return aggregate_metrics(metrics_list)


@torch.no_grad()
def test(opt, model, checkpoint_path, seed):
    seed_everything(seed)
    opt.trainsize = FIXED_IMAGE_SIZE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    load_model_weights(model, checkpoint)
    model.eval()

    datasets = TEST_DATASETS if opt.test_dataset == 'all' else [opt.test_dataset]
    results  = {}

    col = (f'{"Dataset":<26} {"Dice":>6} {"IoU":>6} {"NSD":>6} {"Prec":>6} '
           f'{"Rec":>6} {"Spec":>6} {"BalAcc":>7} {"MCC":>7} {"AUC":>7} '
           f'{"MAE":>6} {"Fw":>6} {"Sm":>6} {"Em":>6} {"HD95":>7}')
    sep = '=' * len(col)
    print(f'\n{sep}\nTEST RESULTS (seed={seed})\n{sep}')
    print(col)
    print(sep)

    for ds in datasets:
        m = _eval_full(model, opt.test_path, ds, opt.trainsize, device)
        results[ds] = m
        print(f'{ds:<26} {m["dice"]:>6.4f} {m["iou"]:>6.4f} {m["nsd"]:>6.4f} '
              f'{m["precision"]:>6.4f} {m["recall"]:>6.4f} {m["specificity"]:>6.4f} '
              f'{m["balanced_acc"]:>7.4f} {m["mcc"]:>7.4f} {m["auc_roc"]:>7.4f} '
              f'{m["mae"]:>6.4f} {m["fw"]:>6.4f} {m["sm"]:>6.4f} '
              f'{m["em"]:>6.4f} {m["hd95"]:>7.2f}')
    print(sep)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed
# ─────────────────────────────────────────────────────────────────────────────

def multiseed_train_test(opt, model_fn, seeds=(42, 123, 456, 789, 2024)):
    opt.trainsize = FIXED_IMAGE_SIZE
    all_results       = {}
    efficiency_report = None

    for seed in seeds:
        print(f'\n{"#"*70}\nSEED {seed}\n{"#"*70}')
        seed_everything(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_fn().to(device)

        if efficiency_report is None:
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            efficiency_report = full_efficiency_report(
                model, input_size=(1, 3, opt.trainsize, opt.trainsize),
                device=device_str)

            # Defense-in-depth on top of the cleanup already done inside
            # full_efficiency_report(): the report runs ~350+ benchmark
            # forward passes plus a deepcopy+thop profiling pass on this
            # exact `model` instance right before it gets handed to
            # train(). thop's hooks create module<->closure reference
            # cycles that plain refcounting can't collect, so without an
            # explicit gc.collect() + empty_cache() here the very first
            # training batch can OOM even though the model itself only
            # needs a few hundred MB.
            gc.collect()
            if device_str == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            model.train()  # measure_* helpers leave the model in eval() mode

        best_path, history = train(opt, model, seed)

        history_path = os.path.join(opt.save_path, f'history_seed{seed}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        results = test(opt, model, best_path, seed)
        all_results[seed] = results

    datasets = TEST_DATASETS if opt.test_dataset == 'all' else [opt.test_dataset]
    summary  = multiseed_stats(all_results)
    print_multiseed_report(summary)

    print('\nLaTeX Table:')
    print(latex_table(summary,
                      caption='Results on polyp segmentation benchmarks.',
                      label=f'tab:{opt.model}'))

    out_path = os.path.join(opt.save_path, f'{opt.model}_multiseed.json')
    with open(out_path, 'w') as f:
        json.dump({
            'seeds':   list(seeds),
            'summary': {
                ds: {k: {kk: float(vv) if not isinstance(vv, list) else vv
                         for kk, vv in v.items()}
                     for k, v in summary[ds].items()}
                for ds in summary
            },
            'raw': {str(s): {ds: all_results[s][ds] for ds in datasets}
                    for s in seeds},
            'efficiency': {
                k: (float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in efficiency_report.items() if v is not None
            },
        }, f, indent=2)
    print(f'\nSaved: {out_path}')
    return summary
