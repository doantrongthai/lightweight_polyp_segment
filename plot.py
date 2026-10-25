import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASETS = ['val', 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
COLORS   = ['#1abc9c', '#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#34495e']
LINES    = ['-', '-', '--', '-.', ':', '-', '--']
METRICS = ['dice', 'iou', 'nsd', 'mae', 'sm', 'em', 'hd95']


def plot_history(history, save_dir, model_name, seed):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, ds in enumerate(DATASETS):
        vals = history.get(ds, [])
        if not vals:
            continue
        epochs = list(range(1, len(vals) + 1))
        ax.plot(epochs, vals,
                color=COLORS[i % len(COLORS)],
                linestyle=LINES[i % len(LINES)],
                linewidth=1.6, label=ds, marker='o', markersize=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.25)

    path = os.path.join(save_dir, f'{model_name}_seed{seed}_curve.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_multiseed_bar(json_path, save_dir):
    with open(json_path) as f:
        data = json.load(f)

    summary  = data['summary']
    datasets = list(summary.keys())
    short    = [d.replace('CVC-', '').replace('ETIS-LaribPolypDB', 'ETIS') for d in datasets]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 5))
    x = np.arange(len(datasets))

    for ax, metric in zip(axes, METRICS):
        means = [summary[ds][metric]['mean'] for ds in datasets]
        stds  = [summary[ds][metric]['std']  for ds in datasets]

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[COLORS[i % len(COLORS)] for i in range(len(datasets))],
                      alpha=0.82, edgecolor='black', linewidth=0.5)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.004,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(metric.upper(), fontsize=9)
        ax.grid(True, axis='y', alpha=0.25)

    path = os.path.join(save_dir, 'multiseed_bar.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_per_seed_dice(json_path, save_dir):
    with open(json_path) as f:
        data = json.load(f)

    seeds    = data['seeds']
    raw      = data['raw']
    datasets = list(data['summary'].keys())
    short    = [d.replace('CVC-', '').replace('ETIS-LaribPolypDB', 'ETIS') for d in datasets]

    x       = np.arange(len(datasets))
    n       = len(seeds)
    width   = 0.8 / n
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, seed in enumerate(seeds):
        dices = [raw[str(seed)][ds]['dice'] for ds in datasets]
        ax.bar(x + offsets[i], dices, width=width,
               color=COLORS[i % len(COLORS)], alpha=0.82,
               label=f'seed={seed}', edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Dice')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=n)
    ax.grid(True, axis='y', alpha=0.25)

    path = os.path.join(save_dir, 'per_seed_dice.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_radar(json_path, save_dir):
    with open(json_path) as f:
        data = json.load(f)

    summary  = data['summary']
    datasets = list(summary.keys())
    radar_metrics = ['dice', 'iou', 'sm', 'em', 'fw']

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})

    for i, ds in enumerate(datasets):
        vals = [summary[ds][m]['mean'] for m in radar_metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[i % len(COLORS)],
                linewidth=1.5, label=ds.replace('CVC-', '').replace('ETIS-LaribPolypDB', 'ETIS'))
        ax.fill(angles, vals, color=COLORS[i % len(COLORS)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in radar_metrics], fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    path = os.path.join(save_dir, 'radar.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True,
                        help='Path to multiseed JSON from trainer')
    parser.add_argument('--save_dir', type=str, default='./plots')
    parser.add_argument('--history', type=str, default=None,
                        help='Path to single-seed history JSON')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.history:
        with open(args.history) as f:
            hist = json.load(f)
        plot_history(hist, args.save_dir, args.model, args.seed)

    plot_multiseed_bar(args.json, args.save_dir)
    plot_per_seed_dice(args.json, args.save_dir)
    plot_radar(args.json, args.save_dir)


if __name__ == '__main__':
    main()
