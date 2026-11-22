"""Render the twin, build the baselines, train every configuration and evaluate it.

    python scripts/run_experiments.py --kvasir kvasir-seg.zip --gastrovision GastroVision.zip --out results

Each trained model is cached as <out>/runs/<config>__<arch>__s<seed>.npz, so an interrupted run
continues where it stopped. All tables are rebuilt at the end (scripts/make_tables.py).
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables
from gastrotwin.baselines import color_transfer_set, cut_paste
from gastrotwin.config import DEVICE, PAPER, QUICK
from gastrotwin.data import extract_zip, load_data
from gastrotwin.model import (audit_detections, build_model, dice_score, image_score, predict, to_tensor,
                              train)
from gastrotwin.stats import ensemble, load_runs
from gastrotwin.twin import ABLATIONS, apply_ablation, make_audit_set, make_twin, render, sample_params

SAVED_WEIGHTS = {('A_kvasir', 'unet_r34', 0), ('C_kvasir+twin', 'unet_r34', 0)}


def prepare(path, dst):
    """Accept either an extracted dataset folder or its zip file."""
    if os.path.isdir(path):
        return path
    if not os.path.isdir(dst):
        print('extracting', path)
        extract_zip(path, dst)
    return dst


def concat(*parts):
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def experiment_plan(cfg, kv, twin, twin3k, twin1k, cut, colour, ablated):
    r34, mit, seeds, abl = 'unet_r34', 'unet_mitb2', cfg.seeds, cfg.seeds_ablation
    return [
        ('A_kvasir', lambda: kv, r34, seeds),
        ('D_kvasir+colour', lambda: concat(kv, colour), r34, seeds),
        ('E_kvasir+cutpaste', lambda: concat(kv, cut), r34, seeds),
        ('B_twin', lambda: twin, r34, seeds),
        ('C3k_kvasir+twin3k', lambda: concat(kv, twin3k), r34, seeds),
        ('C_kvasir+twin', lambda: concat(kv, twin), r34, seeds),
        ('F_kvasir+twin+cutpaste', lambda: concat(kv, twin, cut), r34, seeds),
        ('C1k_kvasir+twin1k', lambda: concat(kv, twin1k), r34, abl),
        *[(f'C3k-{a}', (lambda a=a: concat(kv, ablated[a])), r34, abl) for a in ABLATIONS],
        ('A_kvasir', lambda: kv, mit, seeds),
        ('E_kvasir+cutpaste', lambda: concat(kv, cut), mit, seeds),
        ('C_kvasir+twin', lambda: concat(kv, twin), mit, seeds),
        ('C1k_kvasir+twin1k', lambda: concat(kv, twin1k), mit, seeds),
        ('C3k_kvasir+twin3k', lambda: concat(kv, twin3k), mit, seeds),
    ]


def figure_twin_samples(ref_col, out):
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(3, 6, figsize=(15, 8))
    for i in range(9):
        g = sample_params(rng, ref_col, polyp=True)
        im, m = render(g)
        ax.flat[2 * i].imshow(im)
        ax.flat[2 * i].set_title(f"{g['ptype']} {g['diam']:.0f} mm {g['view']:.0f}°", fontsize=8)
        ax.flat[2 * i + 1].imshow(m, cmap='gray')
    for a in ax.flat:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(f'{out}/fig_twin_samples.png', dpi=150)
    plt.close(fig)


def figure_ablations(ref_col, out):
    cols = [None, *ABLATIONS]
    fig, ax = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 8))
    for r, seed in enumerate([3, 7, 11]):
        for c, a in enumerate(cols):
            g = sample_params(np.random.default_rng(seed), ref_col, polyp=True)
            g = apply_ablation(g, a, np.random.default_rng(seed + 1))
            ax[r, c].imshow(render(g)[0])
            ax[r, c].axis('off')
            ax[r, c].set_title(a or 'full twin', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{out}/fig_ablation_examples.png', dpi=150)
    plt.close(fig)


def kid(a, b, n=100, seed=0):
    """Kernel inception distance with a cubic polynomial kernel (unbiased MMD^2), mean and SD over subsets."""
    rng = np.random.default_rng(seed)
    m, d = min(50, len(a), len(b)), a.shape[1]
    kernel = lambda x, y: (x @ y.T / d + 1) ** 3
    vals = []
    for _ in range(n):
        x, y = a[rng.choice(len(a), m, False)], b[rng.choice(len(b), m, False)]
        kxx, kyy = kernel(x, x), kernel(y, y)
        vals.append((kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kernel(x, y).mean())
    return np.mean(vals), np.std(vals)


def domain_gap(sets, data, cfg, out):
    enc = smp.encoders.get_encoder('resnet34', weights='imagenet' if cfg.pretrained else None).to(DEVICE).eval()

    @torch.no_grad()
    def feats(x, bs=128):
        return np.concatenate([enc(to_tensor(x[i:i + bs]).to(DEVICE))[-1].mean((2, 3)).cpu().numpy()
                               for i in range(0, len(x), bs)])

    def subset(x, n=500):
        return x[np.random.default_rng(0).permutation(len(x))[:n]]

    pos, neg = np.where(data.gl == 1)[0], np.where(data.gl == 0)[0]
    ft = {k: feats(subset(v)) for k, v in sets.items()}
    f_pos, f_norm = feats(data.gx[pos]), feats(subset(data.gx[neg]))
    table = pd.DataFrame([{'set': k, 'KID to gastric polyps': '%.4f ± %.4f' % kid(v, f_pos),
                           'KID to normal stomach': '%.4f ± %.4f' % kid(v, f_norm)} for k, v in ft.items()])
    table.to_csv(f'{out}/table5_domain_gap_kid.csv', index=False)
    print(table.to_string(index=False))

    show = ['Kvasir-SEG (colon)', 'Cut-paste (E)', 'Twin (full)']
    z = np.concatenate([ft[k][:200] for k in show] + [f_norm[:200], f_pos])
    labels = np.array(sum([[k] * len(ft[k][:200]) for k in show], [])
                      + ['GastroVision normal'] * len(f_norm[:200]) + ['GastroVision gastric polyp'] * len(f_pos))
    emb = TSNE(2, perplexity=min(30, len(z) // 4), random_state=0).fit_transform(z)
    fig = plt.figure(figsize=(7, 6))
    for k in dict.fromkeys(labels):
        plt.scatter(*emb[labels == k].T, s=30 if k == 'GastroVision gastric polyp' else 6, label=k, alpha=0.8)
    plt.legend(fontsize=8)
    plt.axis('off')
    plt.title('t-SNE of ImageNet ResNet-34 features')
    fig.tight_layout()
    fig.savefig(f'{out}/fig_domain_gap_tsne.png', dpi=150)
    plt.close(fig)


def figure_gastric_predictions(data, out, pretrained):
    def load(name):
        m = build_model('unet_r34', pretrained).to(DEVICE)
        m.load_state_dict(torch.load(f'{out}/{name}__unet_r34.pt', map_location=DEVICE))
        return m.eval()

    runs = load_runs(f'{out}/runs')
    pos = np.where(data.gl == 1)[0]
    score = ensemble(runs, ('C_kvasir+twin', 'unet_r34'))
    idx = pos[np.argsort(-score[pos])][:8]
    preds = [predict(load(n), data.gx[idx]) for n in ['A_kvasir', 'C_kvasir+twin']]
    fig, ax = plt.subplots(3, len(idx), figsize=(2.2 * len(idx), 7))
    for j, i in enumerate(idx):
        ax[0, j].imshow(data.gx[i])
        for r, p in enumerate(preds, 1):
            ax[r, j].imshow(data.gx[i])
            ax[r, j].imshow(p[j], alpha=0.5, cmap='jet', vmin=0, vmax=1)
    for r, t in enumerate(['GastroVision gastric polyp', 'A: Kvasir only', 'C: Kvasir + twin']):
        ax[r, 0].set_ylabel(t, fontsize=9)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(f'{out}/fig_gastric_predictions.png', dpi=150)
    plt.close(fig)


def main(args):
    cfg = QUICK if args.quick else PAPER
    out = args.out
    os.makedirs(f'{out}/runs', exist_ok=True)
    print(f'device {DEVICE}, torch {torch.__version__}, smp {smp.__version__}, {cfg}')

    data = load_data(prepare(args.kvasir, f'{args.data_dir}/kvasir'),
                     prepare(args.gastrovision, f'{args.data_dir}/gastrovision'))
    print(f'Kvasir-SEG: {len(data.kv_train)} train / {len(data.kv_test)} test | GastroVision test: '
          f'{int(data.gl.sum())} polyps / {int((data.gl == 0).sum())} normal | {len(data.ref_x)} colour references')

    labels = pd.DataFrame({'label': data.gl.astype(int),
                           'class': ['gastric polyp'] * int(data.gl.sum()) + list(data.neg_classes)})
    labels.to_csv(f'{out}/test_labels.csv', index=False)

    t0 = time.time()
    tx, ty, tparams = make_twin(cfg.n_twin, data.ref_col)
    tparams.to_csv(f'{out}/twin_params.csv', index=False)
    print(f'{cfg.n_twin} twin frames in {time.time() - t0:.0f} s')
    ablated = {a: make_twin(cfg.n_syn, data.ref_col, ablation=a)[:2] for a in ABLATIONS}
    colour = color_transfer_set(data.kx, data.ky, data.kv_train, data.ref_x)
    cut = cut_paste(data.kx, data.ky, data.kv_train, data.ref_x, cfg.n_syn)
    ax_, ay_, ameta = make_audit_set(data.ref_col, cfg.audit_reps)
    ameta.to_csv(f'{out}/audit_meta.csv', index=False)
    print(f'{len(ax_)} audit frames')

    figure_twin_samples(data.ref_col, out)
    figure_ablations(data.ref_col, out)

    kv = (data.kx[data.kv_train], data.ky[data.kv_train])
    plan = experiment_plan(cfg, kv, (tx, ty), (tx[:cfg.n_syn], ty[:cfg.n_syn]),
                           (tx[:cfg.n_c1k], ty[:cfg.n_c1k]), cut, colour, ablated)
    n_models = sum(len(seeds) for *_, seeds in plan)
    done, t0 = 0, time.time()
    for name, build_data, arch, seeds in plan:
        for seed in seeds:
            done += 1
            f = f'{out}/runs/{name}__{arch}__s{seed}.npz'
            if os.path.exists(f):
                continue
            x, y = build_data()
            print(f'[{done}/{n_models}] {name} {arch} seed {seed}: {len(x)} images ({(time.time() - t0) / 60:.0f} min)')
            model = train(x, y, seed, arch, cfg)
            np.savez(f, gscore=image_score(predict(model, data.gx)),
                     kdice=dice_score(predict(model, data.kx[data.kv_test]), data.ky[data.kv_test]),
                     audit=audit_detections(predict(model, ax_), ay_))
            if (name, arch, seed) in SAVED_WEIGHTS:
                torch.save(model.state_dict(), f'{out}/{name}__{arch}.pt')
            del model, x, y
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

    json.dump(dict(n_gastric_polyps=int(data.gl.sum()), n_normal_test=int((data.gl == 0).sum()),
                   normal_classes=pd.Series(data.neg_classes).value_counts().to_dict(),
                   n_colour_refs=len(data.ref_x), n_kvasir_train=len(data.kv_train), n_kvasir_test=len(data.kv_test),
                   n_twin=cfg.n_twin, n_syn=cfg.n_syn, iters=cfg.iters, batch=cfg.batch_size,
                   seeds=list(cfg.seeds), seeds_ablation=list(cfg.seeds_ablation), n_audit=len(ax_),
                   torch=torch.__version__, smp=smp.__version__, device=DEVICE),
              open(f'{out}/run_info.json', 'w'), indent=1, default=int)

    sets = {'Kvasir-SEG (colon)': kv[0], 'Colour transfer (D)': colour[0], 'Cut-paste (E)': cut[0], 'Twin (full)': tx,
            **{f'Twin {a}': ablated[a][0] for a in ABLATIONS}}
    domain_gap(sets, data, cfg, out)
    figure_gastric_predictions(data, out, cfg.pretrained)
    make_tables.main(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kvasir', required=True, help='Kvasir-SEG zip or extracted folder')
    parser.add_argument('--gastrovision', required=True, help='GastroVision zip or extracted folder')
    parser.add_argument('--data-dir', default='data', help='where zips are extracted')
    parser.add_argument('--out', default='results')
    parser.add_argument('--quick', action='store_true', help='tiny smoke test, not the paper settings')
    main(parser.parse_args())
