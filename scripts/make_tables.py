"""Build every table and the statistics figures from the per-model score files.

    python scripts/make_tables.py --results results

Needs only results/runs/*.npz (shipped with the repository), so the paper's numbers can be
checked without retraining. Table 4 and the audit tables also need test_labels.csv and
audit_meta.csv, which run_experiments.py writes next to the runs.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from gastrotwin.stats import (METRICS, bootstrap_ci, bootstrap_indices, ensemble, format_p, holm,
                              load_runs, paired_test)
from gastrotwin.twin import ABLATIONS

R34, MIT = 'unet_r34', 'unet_mitb2'
COMPARISONS = [
    *[(c, 'A_kvasir', R34) for c in ['D_kvasir+colour', 'E_kvasir+cutpaste', 'B_twin', 'C1k_kvasir+twin1k',
                                     'C3k_kvasir+twin3k', 'C_kvasir+twin', 'F_kvasir+twin+cutpaste']],
    ('C3k_kvasir+twin3k', 'E_kvasir+cutpaste', R34),
    ('C_kvasir+twin', 'E_kvasir+cutpaste', R34),
    ('F_kvasir+twin+cutpaste', 'E_kvasir+cutpaste', R34),
    *[(c, 'A_kvasir', MIT) for c in ['E_kvasir+cutpaste', 'C1k_kvasir+twin1k', 'C3k_kvasir+twin3k', 'C_kvasir+twin']],
    ('C3k_kvasir+twin3k', 'E_kvasir+cutpaste', MIT),
    ('C_kvasir+twin', 'E_kvasir+cutpaste', MIT),
]
AUDIT_MODELS = [('A_kvasir', R34), ('E_kvasir+cutpaste', R34), ('C_kvasir+twin', R34)]


def mean_sd(values):
    return f'{np.mean(values):.3f} ± {np.std(values):.3f}'


def test_labels(results):
    path = f'{results}/test_labels.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df['label'].to_numpy(float), df.loc[df['label'] == 0, 'class'].to_numpy()
    info = json.load(open(f'{results}/run_info.json'))
    return np.r_[np.ones(info['n_gastric_polyps']), np.zeros(info['n_normal_test'])], None


def table_main(runs, labels, boot):
    rows = []
    for (name, arch), rs in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        sc = ensemble(runs, (name, arch))
        row = {'config': name, 'arch': arch, 'seeds': len(rs),
               'AUROC (seed mean ± SD)': mean_sd([roc_auc_score(labels, r['gscore']) for r in rs])}
        for metric, fn in METRICS.items():
            lo, hi = bootstrap_ci(fn, labels, sc, boot)
            row[f'{metric} ensemble [95% CI]'] = f'{fn(labels, sc):.3f} [{lo:.3f}, {hi:.3f}]'
        row['Kvasir Dice'] = mean_sd([r['kdice'] for r in rs])
        row['Twin-audit detection'] = mean_sd([r['audit'].mean() for r in rs])
        rows.append(row)
    return pd.DataFrame(rows)


def table_paired(runs, labels, boot):
    rows = []
    for a, b, arch in COMPARISONS:
        if (a, arch) in runs and (b, arch) in runs:
            d, (lo, hi), p = paired_test(labels, ensemble(runs, (a, arch)), ensemble(runs, (b, arch)), boot)
            rows.append({'comparison': f'{a} vs {b}', 'arch': arch, 'dAUROC': f'{d:+.3f}',
                         '95% CI': f'[{lo:+.3f}, {hi:+.3f}]', 'p': format_p(p)})
    return pd.DataFrame(rows)


def table_ablation(runs, labels, boot):
    """Each ablated twin against the full twin (C3k), on the same seeds, Holm-corrected."""
    full = ('C3k_kvasir+twin3k', R34)
    rows = []
    for a in ABLATIONS:
        key = (f'C3k-{a}', R34)
        if key not in runs or full not in runs:
            continue
        seeds = {r['seed'] for r in runs[key]}
        d, (lo, hi), p = paired_test(labels, ensemble(runs, key, seeds), ensemble(runs, full, seeds), boot)
        rows.append({'ablation': a, 'seeds': len(seeds), 'dAUROC': f'{d:+.3f}', '95% CI': f'[{lo:+.3f}, {hi:+.3f}]',
                     'p_raw': p})
    df = pd.DataFrame(rows)
    if len(df):
        df['p_holm'] = [format_p(p) for p in holm(df['p_raw'])]
        df['p_raw'] = df['p_raw'].map(format_p)
    return df


def table_fpr_by_class(runs, labels, neg_classes):
    neg = labels == 0
    cols = {}
    for name, arch in AUDIT_MODELS:
        if (name, arch) in runs:
            sc = ensemble(runs, (name, arch))[neg]
            thr = np.quantile(sc, 0.9)
            cols[f'{name} ({arch})'] = {c: np.mean(sc[neg_classes == c] > thr) for c in np.unique(neg_classes)}
    df = pd.DataFrame(cols).round(3)
    df.insert(0, 'n', pd.Series(neg_classes).value_counts())
    return df


def audit_outputs(runs, meta, out):
    models = [k for k in AUDIT_MODELS if k in runs]
    for name, arch in models:
        meta[name] = np.mean([r['audit'] for r in runs[(name, arch)]], 0)
    meta.to_csv(f'{out}/audit.csv', index=False)
    by_type = meta.groupby('type')[[n for n, _ in models]].mean().round(3)
    by_size = meta.groupby('size')[[n for n, _ in models]].mean().round(3)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2), squeeze=False)
    for ax, (name, _) in zip(axes[0], models):
        hm = meta.pivot_table(index='size', columns='angle', values=name, aggfunc='mean')
        im = ax.imshow(hm.values, vmin=0, vmax=1, cmap='RdYlGn', origin='lower')
        ax.set_xticks(range(len(hm.columns)), hm.columns)
        ax.set_yticks(range(len(hm.index)), hm.index)
        ax.set_xlabel('viewing angle from wall normal (°)')
        ax.set_ylabel('polyp diameter (mm)')
        ax.set_title(name)
        for (r, c), v in np.ndenumerate(hm.values):
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=axes)
    fig.savefig(f'{out}/fig_audit_heatmap.png', dpi=150)
    plt.close(fig)
    return by_type, by_size


def roc_figure(runs, labels, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ['A_kvasir', 'D_kvasir+colour', 'E_kvasir+cutpaste', 'B_twin', 'C3k_kvasir+twin3k', 'C_kvasir+twin']:
        if (name, R34) in runs:
            sc = ensemble(runs, (name, R34))
            fpr, tpr, _ = roc_curve(labels, sc)
            ax.plot(fpr, tpr, label=f'{name} ({roc_auc_score(labels, sc):.3f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
    ax.set_xlabel('1 − specificity')
    ax.set_ylabel('sensitivity')
    ax.set_title('Gastric polyp detection, U-Net ResNet-34 seed ensembles')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{out}/fig_roc.png', dpi=150)
    plt.close(fig)


def main(results):
    runs = load_runs(f'{results}/runs')
    labels, neg_classes = test_labels(results)
    boot = bootstrap_indices(labels)
    print(f'{len(runs)} configurations, {sum(map(len, runs.values()))} models; '
          f'test set {int(labels.sum())} polyps / {int((labels == 0).sum())} normal')

    tables = {'table1_main': table_main(runs, labels, boot),
              'table2_paired_tests': table_paired(runs, labels, boot),
              'table3_ablations': table_ablation(runs, labels, boot)}
    if neg_classes is not None:
        tables['table4_fpr_by_normal_class'] = table_fpr_by_class(runs, labels, neg_classes)
    if os.path.exists(f'{results}/audit_meta.csv'):
        tables['table6_audit_by_type'], tables['table7_audit_by_size'] = audit_outputs(
            runs, pd.read_csv(f'{results}/audit_meta.csv'), results)
    roc_figure(runs, labels, results)

    for name, df in tables.items():
        df.to_csv(f'{results}/{name}.csv', index=name.startswith(('table4', 'table6', 'table7')))
        print(f'\n== {name}\n{df.to_string()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', default='results')
    main(parser.parse_args().results)
