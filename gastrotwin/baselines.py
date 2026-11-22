"""Baselines without the twin: Reinhard colour transfer (D) and Poisson cut-paste (E)."""
import cv2
import numpy as np

from .config import IMG_SIZE as S


def _lab_stats(im):
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg = im.max(-1) > 20
    return lab, lab[fg].mean(0), lab[fg].std(0) + 1e-3


def color_transfer(src, ref):
    """Match the Lab mean and SD of src to ref, ignoring black borders (Reinhard et al. 2001)."""
    lab, mu, sd = _lab_stats(src)
    _, mu_r, sd_r = _lab_stats(ref)
    out = np.clip((lab - mu) / sd * sd_r + mu_r, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def color_transfer_set(kx, ky, train_idx, ref_x, seed=3):
    rng = np.random.default_rng(seed)
    x = np.stack([color_transfer(kx[k], ref_x[rng.integers(len(ref_x))]) for k in train_idx])
    return x, ky[train_idx]


def cut_paste(kx, ky, train_idx, ref_x, n, seed=7):
    """Paste colour-matched Kvasir-SEG polyps into gastric references with Poisson blending."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    while len(xs) < n:
        k = rng.choice(train_idx)
        m = ky[k]
        rows, cols = np.nonzero(m)
        if len(rows) < 30:
            continue
        y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        bg = ref_x[rng.integers(len(ref_x))].copy()
        patch = color_transfer(kx[k], bg)[y0:y1, x0:x1]
        pm = m[y0:y1, x0:x1]

        scale = rng.uniform(0.25, 1.0) * min(1.0, 0.5 * S / max(patch.shape[:2]))
        h, w = max(8, int(patch.shape[0] * scale)), max(8, int(patch.shape[1] * scale))
        patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)
        pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST)
        patch = cv2.copyMakeBorder(patch, 3, 3, 3, 3, cv2.BORDER_REFLECT)
        pm = cv2.copyMakeBorder(pm, 3, 3, 3, 3, cv2.BORDER_CONSTANT)

        py, px = np.nonzero(pm)
        bh, bw = py.max() - py.min() + 1, px.max() - px.min() + 1
        cy = int(rng.integers(bh // 2 + 2, S - bh // 2 - 2))
        cx = int(rng.integers(bw // 2 + 2, S - bw // 2 - 2))
        try:
            out = cv2.seamlessClone(patch, bg, pm * 255, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            continue

        # OpenCV centres the bounding box of the mask on (cx, cy).
        ty, tx = cy - bh // 2 - py.min(), cx - bw // 2 - px.min()
        label = np.zeros((S, S), np.uint8)
        yy, xx = py + ty, px + tx
        inside = (yy >= 0) & (yy < S) & (xx >= 0) & (xx < S)
        label[yy[inside], xx[inside]] = 1
        xs.append(out)
        ys.append(label)
    return np.stack(xs), np.stack(ys)
