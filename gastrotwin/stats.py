"""Seed ensembles, stratified bootstrap confidence intervals, paired tests and Holm correction."""
import glob
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

N_BOOT = 2000


def load_runs(runs_dir):
    """{(config, arch): [{'gscore', 'kdice', 'audit', 'seed'}, ...]} from <config>__<arch>__s<seed>.npz files."""
    runs = {}
    for f in sorted(glob.glob(f'{runs_dir}/*.npz')):
        name, arch, seed = os.path.basename(f)[:-4].split('__')
        runs.setdefault((name, arch), []).append(dict(np.load(f), seed=int(seed[1:])))
    return runs


def ensemble(runs, key, seeds=None):
    return np.mean([r['gscore'] for r in runs[key] if seeds is None or r['seed'] in seeds], 0)


def bootstrap_indices(labels, n=N_BOOT, seed=0):
    """Resample positives and negatives separately; shared by all models so that tests are paired."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    return [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n)]


def sens_at_spec(labels, scores, spec=0.9):
    return np.mean(scores[labels == 1] > np.quantile(scores[labels == 0], spec))


def bootstrap_ci(metric, labels, scores, boot):
    return np.percentile([metric(labels[i], scores[i]) for i in boot], [2.5, 97.5])


def paired_test(labels, score_a, score_b, boot):
    """AUROC(a) - AUROC(b) with bootstrap 95 % CI and a two-sided bootstrap p-value."""
    d = np.array([roc_auc_score(labels[i], score_a[i]) - roc_auc_score(labels[i], score_b[i]) for i in boot])
    delta = roc_auc_score(labels, score_a) - roc_auc_score(labels, score_b)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return delta, np.percentile(d, [2.5, 97.5]), p


def holm(pvalues):
    p = np.asarray(pvalues, float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def format_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


METRICS = {'AUROC': roc_auc_score, 'AUPRC': average_precision_score, 'Sens@Spec90': sens_at_spec}
