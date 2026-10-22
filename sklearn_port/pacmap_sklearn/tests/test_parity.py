"""Parity with the reference (numba) implementation in ``source/pacmap``.

Skipped unless the reference package is importable.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from pacmap_sklearn import PaCMAP
from pacmap_sklearn._pacmap import _incidence_list
from pacmap_sklearn._pacmap_utils import pacmap_gradient

ref = pytest.importorskip("pacmap.pacmap")


@pytest.mark.parametrize(
    "w_nn, w_mn, w_fp", [(2.0, 1000.0, 1.0), (3.0, 3.0, 1.0), (1.0, 0.0, 1.0)]
)
def test_gradient_matches_reference(w_nn, w_mn, w_fp):
    rng = np.random.default_rng(0)
    n = 500
    Y = rng.normal(size=(n, 2)) * 3
    blocks = []
    for k in (10, 5, 20):
        P = rng.integers(0, n, size=(n * k, 2)).astype(np.int32)
        P[:, 0] = np.repeat(np.arange(n), k)
        blocks.append(P)
    expected = ref.pacmap_grad(Y.astype(np.float32), *blocks, w_nn, w_mn, w_fp)

    pairs = np.concatenate(blocks).astype(np.intp)
    types = np.concatenate([np.full(len(P), t, np.int8) for t, P in enumerate(blocks)])
    indptr, other, t = _incidence_list(pairs, types, n)
    grad, node_loss = np.empty_like(Y), np.empty(n)
    pacmap_gradient(Y, indptr, other, t, w_nn, w_mn, w_fp, grad, node_loss, 2)

    scale = np.abs(expected[:-1]).max()
    assert_allclose(grad / scale, expected[:-1] / scale, atol=1e-5)
    assert 0.5 * node_loss.sum() == pytest.approx(expected[-1, 0], rel=1e-5)


def test_localmap_gradient_matches_reference():
    rng = np.random.default_rng(0)
    n = 400
    Y = rng.normal(size=(n, 2)) * 3
    blocks = []
    for k in (10, 5, 20):
        P = rng.integers(0, n, size=(n * k, 2)).astype(np.int32)
        P[:, 0] = np.repeat(np.arange(n), k)
        blocks.append(P)
    expected = ref.pacmap_grad_nearby_recip_sqrt(
        Y.astype(np.float32), *blocks, 1.0, 0.0, 1.0, 5.0
    )
    pairs = np.concatenate(blocks).astype(np.intp)
    types = np.concatenate([np.full(len(P), t, np.int8) for t, P in enumerate(blocks)])
    indptr, other, t = _incidence_list(pairs, types, n)
    grad, node_loss = np.empty_like(Y), np.empty(n)
    pacmap_gradient(Y, indptr, other, t, 1.0, 0.0, 1.0, grad, node_loss, 2, 5.0)
    scale = np.abs(expected[:-1]).max()
    assert_allclose(grad / scale, expected[:-1] / scale, atol=1e-5)


@pytest.mark.parametrize("thres", [2.0, 5.0])
def test_localmap_resampling_statistics_match_reference(thres):
    # Different RNGs, so compare distributions: fraction of slots that keep
    # their old partner and median embedding distance of the new pairs.
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import NearestNeighbors

    from pacmap_sklearn._pacmap_utils import resample_fp_nearby, sample_fp_pairs

    Y, _ = make_blobs(3000, n_features=2, centers=15, cluster_std=1.5,
                      center_box=(-60, 60), random_state=0)
    n = len(Y)
    nbrs = NearestNeighbors(n_neighbors=11).fit(Y).kneighbors(return_distance=False)
    nbrs = np.ascontiguousarray(nbrs[:, 1:], dtype=np.intp)
    pnn = np.column_stack([np.repeat(np.arange(n), 10), nbrs.ravel()]).astype(np.int32)
    old = sample_fp_pairs(nbrs, 20, 1, 2)

    def stats(new):
        d = np.linalg.norm(Y[new[:, 0]] - Y[new[:, 1]], axis=1)
        return (new[:, 1] == old[:, 1]).mean(), np.median(d)

    Y32 = Y.astype(np.float32)
    r = np.mean([stats(ref.sample_FP_pair_nearby(Y32, pnn, old.astype(np.int32), Y32, thres))
                 for _ in range(3)], axis=0)
    p = np.mean([stats(resample_fp_nearby(Y, nbrs, old, thres, s, 0, 2))
                 for s in range(3)], axis=0)
    assert abs(r[0] - p[0]) < 0.01
    assert abs(r[1] - p[1]) / r[1] < 0.02


def _scores(X, y, emb):
    return (
        trustworthiness(X, emb, n_neighbors=10),
        cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean(),
    )


def test_embedding_quality_matches_reference():
    X, y = load_digits(return_X_y=True)
    ports, refs = [], []
    for seed in range(3):
        ports.append(_scores(X, y, PaCMAP(random_state=seed).fit_transform(X)))
        refs.append(
            _scores(X, y, ref.PaCMAP(random_state=seed, knn_backend="faiss").fit_transform(X))
        )
    port_tw, port_acc = np.mean(ports, axis=0)
    ref_tw, ref_acc = np.mean(refs, axis=0)
    assert port_tw >= ref_tw - 0.005
    assert port_acc >= ref_acc - 0.01


def test_localmap_quality_matches_reference():
    from pacmap_sklearn import LocalMAP

    X, y = load_digits(return_X_y=True)
    ports = [_scores(X, y, LocalMAP(random_state=s).fit_transform(X)) for s in range(3)]
    refs = [_scores(X, y, ref.LocalMAP(random_state=s).fit_transform(X)) for s in range(3)]
    port_tw, port_acc = np.mean(ports, axis=0)
    ref_tw, ref_acc = np.mean(refs, axis=0)
    assert port_tw >= ref_tw - 0.005
    assert port_acc >= ref_acc - 0.01
