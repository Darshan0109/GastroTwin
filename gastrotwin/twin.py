"""Stomach digital twin: a sphere-traced signed-distance scene of the gastric lumen.

The lumen has rugal folds, the polyp is one of the Paris types 0-Ip, 0-Is or 0-IIa, and the
gastroscope is a fisheye camera with a co-located light. Every frame comes with an exact mask.
Lengths are in mm.
"""
import math

import cv2
import numpy as np
import pandas as pd
import torch

from .config import DEVICE, IMG_SIZE

ABLATIONS = {
    'no_folds': dict(fold_a=0.0),
    'no_specular': dict(ks=0.0),
    'no_lobulation': dict(lobe=0.0),
    'no_polyp_tint': dict(tint=np.ones(3)),
    'no_color_calib': 'random',
}
SCENE_KEYS = ['polyp', 'ptype', 'diam', 'view', 'dist_eff']
TRACE_STEPS = 128


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _t(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def wall_radius(z, theta, g):
    base = g['R0'] * (1 + 0.2 * torch.sin(0.03 * z + g['ph']))
    folds = torch.sin(g['fold_k'] * theta + 0.4 * torch.sin(0.05 * z)).clamp_min(0) ** 2
    return base - g['fold_a'] * folds


def sdf_wall(p, g):
    r = wall_radius(p[..., 2], torch.atan2(p[..., 1], p[..., 0]), g)
    return (r - p[..., :2].norm(dim=-1)) * 0.6


def sdf_polyp(p, g):
    w, n, t1, t2, r = g['W'], g['n'], g['t1'], g['t2'], g['diam'] / 2
    if g['ptype'] == 'Is':
        return (p - (w + n * 0.3 * r)).norm(dim=-1) - r
    if g['ptype'] == 'IIa':
        q = p - w
        h = 0.35 * r
        k = ((q @ n / h) ** 2 + (q @ t1 / r) ** 2 + (q @ t2 / r) ** 2).sqrt()
        return (k - 1) * h
    head = w + n * 1.8 * r
    a = w - n * r
    ba = head - a
    s = (((p - a) @ ba) / (ba @ ba)).clamp(0, 1)
    stalk = (p - a - s[..., None] * ba).norm(dim=-1) - 0.3 * r
    return torch.minimum((p - head).norm(dim=-1) - r, stalk)


def scene(p, g):
    """Signed distances to the wall and to the polyp."""
    w = sdf_wall(p, g)
    if not g['polyp']:
        return w, torch.full_like(w, 1e3)
    r = g['diam'] / 2
    f = 4 / r
    bump = 0.12 * r * torch.sin(f * p[..., 0] + g['ph']) * torch.sin(f * p[..., 1]) * torch.sin(f * p[..., 2])
    return w, sdf_polyp(p, g) * 0.8 + g.get('lobe', 1.0) * bump


def sample_params(rng, ref_col, **overrides):
    """Draw one scene. The draw order defines the dataset, so do not reorder these lines."""
    g = dict(
        R0=rng.uniform(25, 45), ph=rng.uniform(0, 6.3), fold_a=rng.uniform(0.5, 3.5),
        fold_k=int(rng.integers(6, 14)),
        polyp=rng.random() < 0.75, ptype=str(rng.choice(['Ip', 'Is', 'IIa'])), diam=rng.uniform(3, 20),
        view=rng.uniform(0, 75), dist=rng.uniform(15, 60), off=rng.uniform(0, 0.6), roll=rng.uniform(0, 6.3),
        fov=math.radians(rng.uniform(120, 150)), shin=rng.uniform(20, 80), ks=rng.uniform(0.3, 1.2),
        tex=rng.uniform(0.03, 0.12), col=ref_col[rng.integers(len(ref_col))] * rng.uniform(0.9, 1.1, 3),
        tint=np.array([rng.uniform(0.95, 1.15), rng.uniform(0.7, 1.0), rng.uniform(0.7, 1.0)]),
        blur=rng.uniform(0, 1.5), noise=rng.uniform(0, 0.02), seed=int(rng.integers(1 << 30)),
    )
    g.update(overrides)
    return g


def _camera(g, rng):
    theta, z = rng.uniform(0, 2 * math.pi), rng.uniform(-40, 40)
    radius = wall_radius(_t(z), _t(theta), g).item()
    n = _t([-math.cos(theta), -math.sin(theta), 0])
    t1 = _t([-math.sin(theta), math.cos(theta), 0])
    t2 = _t([0, 0, 1])
    g.update(W=-n * radius + t2 * z, n=n, t1=t1, t2=t2)

    a = math.radians(g['view'])
    side = rng.choice([-1, 1])
    dist = min(g['dist'], 1.6 * radius / max(math.cos(a), 1e-3))
    g['dist_eff'] = dist
    cam = g['W'] + dist * (math.cos(a) * n + side * math.sin(a) * t2)
    target = g['W'] + n * 0.5 * g['diam']
    fwd = _unit(target - cam)
    jitter = _unit(torch.cross(fwd, _t(rng.normal(size=3)), dim=0))
    fwd = _unit(fwd + jitter * math.tan(g['off'] * g['fov'] / 2))
    up0 = t1 if abs(float(fwd @ t2)) > 0.95 else t2
    right = _unit(torch.cross(fwd, up0, dim=0))
    up = torch.cross(right, fwd, dim=0)
    c, s = math.cos(g['roll']), math.sin(g['roll'])
    return cam, fwd, c * right + s * up, -s * right + c * up


@torch.no_grad()
def render(g):
    """Render one frame. Returns (image uint8 HxWx3, mask uint8 HxW); adds W, n, t1, t2, dist_eff to g."""
    rng = np.random.default_rng(g['seed'])
    cam, fwd, right, up = _camera(g, rng)

    lin = torch.linspace(-1, 1, IMG_SIZE, device=DEVICE)
    v, u = torch.meshgrid(lin, lin, indexing='ij')
    rho = (u ** 2 + v ** 2).sqrt()
    ang = rho * g['fov'] / 2
    k = torch.where(rho > 1e-6, torch.sin(ang) / rho.clamp_min(1e-6), torch.ones_like(rho))
    d = _unit(k[..., None] * (u[..., None] * right + v[..., None] * up) + torch.cos(ang)[..., None] * fwd)

    t = torch.full_like(rho, 0.1)
    for _ in range(TRACE_STEPS):
        w, pl = scene(cam + t[..., None] * d, g)
        t = (t + torch.minimum(w, pl).clamp_min(0.01)).clamp_max(300)
    p = cam + t[..., None] * d
    w, pl = scene(p, g)
    hit = (torch.minimum(w, pl) < 0.1 + 0.002 * t) & (rho <= 1)
    mask = hit & (pl < w + 0.05)

    eps, f0 = 0.05, torch.minimum(w, pl)
    normal = _unit(torch.stack([torch.minimum(*scene(p + eps * ax, g)) - f0
                                for ax in torch.eye(3, device=DEVICE)], -1))
    ndl = (normal * -d).sum(-1).clamp_min(0)
    falloff = 1 / (1 + (t / 25) ** 2)
    tex = 1 + g['tex'] * (torch.sin(0.6 * p[..., 0] + 0.4 * p[..., 2]) * torch.sin(0.7 * p[..., 1] - 0.5 * p[..., 2])
                          + torch.sin(2.1 * p[..., 2] + 1.3 * p[..., 0]))
    tint = torch.where(mask[..., None], _t(g['tint'])[None, None], _t([1, 1, 1])[None, None])
    albedo = _t(g['col'])[None, None] * tint
    rgb = (albedo * tex[..., None] * (0.05 + ndl[..., None]) + g['ks'] * (ndl ** g['shin'])[..., None]) * falloff[..., None]
    rgb = rgb * hit[..., None]

    lum = rgb.mean(-1)[hit]
    rgb = rgb * (0.85 / lum.quantile(0.95).clamp_min(1e-4) if lum.numel() else 1)
    img = (rgb.clamp(0, 1) ** (1 / 2.2)).cpu().numpy()
    if g['blur'] > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), g['blur'])
    img = np.clip(img + rng.normal(0, g['noise'], img.shape), 0, 1) * (rho <= 1).cpu().numpy()[..., None]
    return (img * 255).astype(np.uint8), mask.cpu().numpy().astype(np.uint8)


def apply_ablation(g, ablation, rng_col):
    if ablation == 'no_color_calib':
        g['col'] = rng_col.uniform(0.05, 0.8, 3)
    elif ablation:
        g.update(ABLATIONS[ablation])
    return g


def make_twin(n, ref_col, seed=42, ablation=None):
    """Render n frames. Ablated sets reuse the scenes of the full set (same seed)."""
    rng, rng_col = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    xs, ys, params = [], [], []
    for _ in range(n):
        g = apply_ablation(sample_params(rng, ref_col), ablation, rng_col)
        im, m = render(g)
        xs.append(im)
        ys.append(m)
        params.append({k: g[k] for k in SCENE_KEYS})
    return np.stack(xs), np.stack(ys), pd.DataFrame(params)


AUDIT_SIZES = (3, 5, 8, 12, 20)
AUDIT_ANGLES = (0, 20, 40, 60, 75)
AUDIT_TYPES = ('Ip', 'Is', 'IIa')


def make_audit_set(ref_col, reps=8, seed=123):
    """Fresh frames on a size x viewing-angle x Paris-type grid, camera aimed at the polyp."""
    rng = np.random.default_rng(seed)
    xs, ys, meta = [], [], []
    for ptype in AUDIT_TYPES:
        for size in AUDIT_SIZES:
            for angle in AUDIT_ANGLES:
                for _ in range(reps):
                    im, m = render(sample_params(rng, ref_col, polyp=True, ptype=ptype, diam=size, view=angle, off=0.0))
                    if m.sum() == 0:
                        continue
                    xs.append(im)
                    ys.append(m)
                    meta.append(dict(type=ptype, size=size, angle=angle))
    return np.stack(xs), np.stack(ys), pd.DataFrame(meta)
