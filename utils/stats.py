import numpy as np
from scipy import stats
from scipy.stats import wilcoxon


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    values = np.array(values)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {'mean': float(values.mean()), 'ci_low': lo, 'ci_high': hi, 'std': float(values.std())}


def wilcoxon_test(scores_a, scores_b, alternative='two-sided'):
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
    stat, p = wilcoxon(diff, alternative=alternative)
    return {'statistic': float(stat), 'p_value': float(p), 'significant': bool(p < 0.05)}


def _delong_auc_variance(y_true, y_score):
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    npos = y_true.sum()
    nneg = len(y_true) - npos
    if npos == 0 or nneg == 0:
        return 0.5, 0.0

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    tp_prev = np.concatenate([[0], tp[:-1]])
    fp_prev = np.concatenate([[0], fp[:-1]])

    auc = float(np.sum((fp - fp_prev) * (tp + tp_prev) / 2) / (npos * nneg))

    v10 = np.zeros(int(npos))
    v01 = np.zeros(int(nneg))
    pos_idx = 0
    neg_idx = 0
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            v10[pos_idx] = (fp[i] - (fp_prev[i] if i > 0 else 0)) / nneg
            pos_idx += 1
        else:
            v01[neg_idx] = (tp[i] - (tp_prev[i] if i > 0 else 0)) / npos
            neg_idx += 1

    var = (np.var(v10) / npos + np.var(v01) / nneg)
    return auc, float(var)


def delong_test(y_true, scores_a, scores_b):
    y_true = np.array(y_true)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    auc_a, var_a = _delong_auc_variance(y_true, scores_a)
    auc_b, var_b = _delong_auc_variance(y_true, scores_b)

    diff = auc_a - auc_b
    se = np.sqrt(var_a + var_b + 1e-12)
    z = diff / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        'auc_a': float(auc_a),
        'auc_b': float(auc_b),
        'diff':  float(diff),
        'z':     float(z),
        'p_value': p,
        'significant': bool(p < 0.05),
    }


def holm_bonferroni(p_values, alpha=0.05):
    p_values = np.array(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = alpha / np.arange(n, 0, -1)
    reject = np.zeros(n, dtype=bool)
    for i in range(n):
        if sorted_p[i] <= thresholds[i]:
            reject[order[i]] = True
        else:
            break
    adjusted = np.minimum(1.0, sorted_p * np.arange(n, 0, -1))
    adjusted = np.maximum.accumulate(adjusted)
    adj_out = np.zeros(n)
    adj_out[order] = adjusted
    return {'reject': reject.tolist(), 'adjusted_p': adj_out.tolist()}


def multiseed_stats(results_per_seed, metric_keys=None):
    if metric_keys is None:
        first = list(results_per_seed.values())[0]
        metric_keys = list(list(first.values())[0].keys())

    datasets = list(list(results_per_seed.values())[0].keys())
    seeds = list(results_per_seed.keys())
    summary = {}

    for ds in datasets:
        summary[ds] = {}
        for k in metric_keys:
            vals = [results_per_seed[s][ds][k] for s in seeds]
            ci = bootstrap_ci(vals)
            summary[ds][k] = {
                'mean':     ci['mean'],
                'std':      ci['std'],
                'ci_low':   ci['ci_low'],
                'ci_high':  ci['ci_high'],
                'values':   vals,
            }
    return summary


def latex_table(summary, metrics=('dice', 'iou', 'mae', 'sm', 'em', 'hd95', 'nsd'),
                caption='', label='tab:results'):
    datasets = list(summary.keys())
    short = {
        'CVC-300': 'CVC-300',
        'CVC-ClinicDB': 'ClinicDB',
        'Kvasir': 'Kvasir',
        'CVC-ColonDB': 'ColonDB',
        'ETIS-LaribPolypDB': 'ETIS',
    }
    header_metrics = ' & '.join([m.upper().replace('_', '-') for m in metrics])
    col_spec = 'l' + 'c' * len(metrics)

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    if caption:
        lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')
    lines.append(f'Dataset & {header_metrics} \\\\')
    lines.append(r'\midrule')

    for ds in datasets:
        name = short.get(ds, ds)
        row_parts = [name]
        for m in metrics:
            if m not in summary[ds]:
                row_parts.append('--')
                continue
            d = summary[ds][m]
            mean, std = d['mean'], d['std']
            if m == 'mae' or m == 'hd95':
                cell = f'${mean:.4f}\\pm{std:.4f}$'
            else:
                cell = f'${mean:.4f}\\pm{std:.4f}$'
            row_parts.append(cell)
        lines.append(' & '.join(row_parts) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def print_multiseed_report(summary, metrics=('dice', 'iou', 'mae', 'sm', 'em', 'hd95', 'nsd')):
    datasets = list(summary.keys())
    col_w = 16
    header = f'{"Dataset":<20}' + ''.join([f'{m.upper():>{col_w}}' for m in metrics])
    print(f'\n{"="*len(header)}')
    print(header)
    print(f'{"="*len(header)}')
    for ds in datasets:
        row = f'{ds:<20}'
        for m in metrics:
            if m not in summary[ds]:
                row += f'{"--":>{col_w}}'
                continue
            d = summary[ds][m]
            cell = f'{d["mean"]:.4f}±{d["std"]:.4f}'
            row += f'{cell:>{col_w}}'
        print(row)
    print(f'{"="*len(header)}\n')


def compare_models(results_a, results_b, dataset, metric='dice'):
    scores_a = results_a[dataset][metric]['values']
    scores_b = results_b[dataset][metric]['values']
    w = wilcoxon_test(scores_a, scores_b)
    return {
        'metric': metric,
        'dataset': dataset,
        'mean_a': float(np.mean(scores_a)),
        'mean_b': float(np.mean(scores_b)),
        'delta': float(np.mean(scores_a) - np.mean(scores_b)),
        'wilcoxon_p': w['p_value'],
        'significant': w['significant'],
    }
