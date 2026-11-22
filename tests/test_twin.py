"""Smoke test: the twin renders deterministically and ablations keep the scene fixed.

    python -m pytest tests  (or: python tests/test_twin.py)
"""
import numpy as np

from gastrotwin.twin import ABLATIONS, make_audit_set, make_twin

REF_COL = np.random.default_rng(0).uniform(0.05, 0.6, (20, 3))


def test_render_is_deterministic():
    a, b = make_twin(4, REF_COL), make_twin(4, REF_COL)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0].shape == (4, 256, 256, 3) and a[0].dtype == np.uint8


def test_ablations_reuse_the_scenes():
    full = make_twin(3, REF_COL)[2]
    for name in ABLATIONS:
        assert make_twin(3, REF_COL, ablation=name)[2].equals(full), name


def test_audit_frames_contain_a_polyp():
    _, masks, meta = make_audit_set(REF_COL, reps=1)
    assert len(masks) == len(meta) > 0 and all(m.sum() > 0 for m in masks)


if __name__ == '__main__':
    test_render_is_deterministic()
    test_ablations_reuse_the_scenes()
    test_audit_frames_contain_a_polyp()
    print('ok')
