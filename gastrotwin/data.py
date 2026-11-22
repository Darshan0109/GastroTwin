"""Kvasir-SEG and GastroVision loading, splits and the twin's colour references."""
import glob
import os
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np

from .config import IMG_SIZE, NEG_CLASSES, POS_CLASS

IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def extract_zip(zip_path, dst):
    """Extract a dataset zip, including the nested zips of some GastroVision releases."""
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(zip_path)
    zipfile.ZipFile(zip_path).extractall(dst)
    for inner in glob.glob(f'{dst}/**/*.zip', recursive=True):
        zipfile.ZipFile(inner).extractall(os.path.dirname(inner))
        os.remove(inner)


def load_image(path, mask=False):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
    interp = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=interp)
    return (im > 127).astype(np.uint8) if mask else np.ascontiguousarray(im[..., ::-1])


def _norm(name):
    return name.lower().replace('_', ' ').replace('-', ' ')


def split_gastrovision(root):
    """Return (positive files, held-out negatives as (file, class), colour-reference files).

    30 % of the negatives are colour references for the twin and backgrounds for cut-paste;
    they are never used for testing.
    """
    classes = {}
    for f in glob.glob(f'{root}/**/*', recursive=True):
        if f.lower().endswith(IMAGE_EXT) and '__MACOSX' not in f:
            classes.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    pos = sorted(f for k, v in classes.items() if POS_CLASS in _norm(k) for f in v)
    neg = sorted((f, k) for k, v in classes.items() if any(n in _norm(k) for n in NEG_CLASSES) for f in v)
    if not pos or not neg:
        raise RuntimeError(f'GastroVision class folders not found under {root}: {sorted(classes)}')
    np.random.default_rng(0).shuffle(neg)
    n_ref = int(0.3 * len(neg))
    return pos, neg[n_ref:], [f for f, _ in neg[:n_ref]]


@dataclass
class Data:
    kx: np.ndarray            # Kvasir-SEG images (N, S, S, 3) uint8 RGB
    ky: np.ndarray            # Kvasir-SEG masks (N, S, S) uint8
    kv_train: np.ndarray      # indices of the 800 training images
    kv_test: np.ndarray       # indices of the 200 in-domain test images
    ref_x: np.ndarray         # GastroVision colour references
    ref_col: np.ndarray       # mean linear-RGB mucosa colour of each reference
    gx: np.ndarray            # GastroVision test images, positives first
    gl: np.ndarray            # 1 = gastric polyp, 0 = normal
    neg_classes: np.ndarray   # class name of every negative test image


def load_data(kvasir_root, gastrovision_root):
    kv_dir = os.path.dirname(glob.glob(f'{kvasir_root}/**/images', recursive=True)[0])
    kv_files = sorted(glob.glob(f'{kv_dir}/images/*'))
    kx = np.stack([load_image(f) for f in kv_files])
    ky = np.stack([load_image(f.replace('/images/', '/masks/'), mask=True) for f in kv_files])
    perm = np.random.default_rng(0).permutation(len(kx))
    n_test = len(kx) // 5

    pos, neg_test, ref = split_gastrovision(gastrovision_root)
    ref_x = np.stack([load_image(f) for f in ref])
    ref_col = (ref_x[:, 64:192, 64:192] / 255.).reshape(len(ref_x), -1, 3).mean(1) ** 2.2
    gx = np.stack([load_image(f) for f in pos + [f for f, _ in neg_test]])
    gl = np.r_[np.ones(len(pos)), np.zeros(len(neg_test))]
    return Data(kx, ky, perm[n_test:], perm[:n_test], ref_x, ref_col, gx, gl,
                np.array([k for _, k in neg_test]))
