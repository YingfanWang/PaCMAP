import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from threadpoolctl import threadpool_limits

from pacmap_sklearn import LocalMAP, PaCMAP
from pacmap_sklearn._pacmap_utils import (
    build_incidence,
    resample_fp_nearby,
    sample_fp_pairs,
)

FAST = dict(num_iters=(30, 30, 60))


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def test_build_incidence_matches_stable_argsort():
    rng = np.random.RandomState(0)
    n = 200
    pairs = rng.randint(0, n, size=(3000, 2)).astype(np.intp)
    types = rng.randint(0, 3, 3000).astype(np.int8)
    indptr, other, t = build_incidence(pairs, types, n)

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    order = np.argsort(rows, kind="stable")
    assert_array_equal(indptr, np.r_[0, np.cumsum(np.bincount(rows, minlength=n))])
    assert_array_equal(other, np.concatenate([pairs[:, 1], pairs[:, 0]])[order])
    assert_array_equal(t, np.concatenate([types, types])[order])


@pytest.fixture(scope="module")
def blob_embedding():
    Y, _ = make_blobs(1500, n_features=2, centers=12, cluster_std=1.5,
                      center_box=(-60, 60), random_state=0)
    nbrs = NearestNeighbors(n_neighbors=11).fit(Y).kneighbors(return_distance=False)
    nbrs = np.ascontiguousarray(nbrs[:, 1:], dtype=np.intp)
    old = sample_fp_pairs(nbrs, 20, 1, 1)
    return Y, nbrs, old


@pytest.mark.parametrize("thres", [2.0, 10.0])
def test_resample_fp_nearby_invariants(blob_embedding, thres):
    Y, nbrs, old = blob_embedding
    n, n_fp = len(Y), old.shape[0] // len(Y)
    new = resample_fp_nearby(Y, nbrs, old, thres, 0, 0, 2)
    assert_array_equal(new[:, 0], old[:, 0])
    replaced = new[:, 1] != old[:, 1]
    j = new[:, 1]
    i = new[:, 0]
    assert np.all(j != i)
    d = np.linalg.norm(Y[i] - Y[j], axis=1)
    assert np.all(d[replaced] <= thres)
    is_nn = np.array([jj in nbrs[ii] for ii, jj in zip(i, j)])
    assert not np.any(is_nn[replaced])
    assert replaced.mean() > 0.5  # most slots find a nearby partner


def test_resample_fp_nearby_thread_independent(blob_embedding):
    Y, nbrs, old = blob_embedding
    a = resample_fp_nearby(Y, nbrs, old, 5.0, 3, 120, 1)
    b = resample_fp_nearby(Y, nbrs, old, 5.0, 3, 120, 2)
    assert_array_equal(a, b)
    c = resample_fp_nearby(Y, nbrs, old, 5.0, 3, 130, 2)
    assert not np.array_equal(a, c)  # rounds use different streams


# ---------------------------------------------------------------------------
# LocalMAP
# ---------------------------------------------------------------------------


def test_localmap_deterministic_and_thread_independent():
    X, _ = make_blobs(500, n_features=20, centers=6, random_state=0)
    with threadpool_limits(limits=1, user_api="openmp"):
        a = LocalMAP(random_state=3, **FAST).fit_transform(X)
    with threadpool_limits(limits=4, user_api="openmp"):
        b = LocalMAP(random_state=3, **FAST).fit_transform(X)
    assert_array_equal(a, b)


def test_localmap_equals_pacmap_before_phase3():
    # Phase-3 behavior starts strictly after n1 + n2, so with n3 = 1 the
    # only phase-3 iteration (itr == n1 + n2) is a plain PaCMAP step.
    X, _ = make_blobs(300, n_features=10, random_state=0)
    kw = dict(num_iters=(20, 20, 1), random_state=0)
    assert_array_equal(
        LocalMAP(**kw).fit_transform(X), PaCMAP(**kw).fit_transform(X)
    )


def test_localmap_resamples_fp_pairs():
    X, _ = make_blobs(400, n_features=10, centers=5, random_state=0)
    kw = dict(num_iters=(20, 20, 30), random_state=0)
    pm, lm = PaCMAP(**kw).fit(X), LocalMAP(**kw).fit(X)
    assert_array_equal(pm.pair_neighbors_, lm.pair_neighbors_)
    assert not np.array_equal(pm.pair_fp_, lm.pair_fp_)
    d = lambda est: np.linalg.norm(
        est.embedding_[est.pair_fp_[:, 0]] - est.embedding_[est.pair_fp_[:, 1]], axis=1
    )
    assert np.median(d(lm)) < np.median(d(pm))


def test_localmap_digits_quality():
    X, y = load_digits(return_X_y=True)
    emb = LocalMAP(random_state=0).fit_transform(X)
    assert cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean() > 0.95


# ---------------------------------------------------------------------------
# Rashomon neighbor refinement
# ---------------------------------------------------------------------------


def test_rashomon_selection():
    X, _ = make_blobs(400, n_features=15, centers=5, random_state=0)
    est = PaCMAP(n_rashomon=3, random_state=0, **FAST).fit(X)
    n, k = X.shape[0], est.n_neighbors_
    cand, psi = est.rashomon_candidates_, est.rashomon_scores_
    assert cand.shape == psi.shape == (n, 50)
    nbrs = est.pair_neighbors_[:, 1].reshape(n, k)
    # The chosen neighbors are exactly the k lowest-score candidates.
    order = np.argsort(psi, axis=1, kind="stable")[:, :k]
    assert_array_equal(nbrs, np.take_along_axis(cand, order, axis=1))
    # Candidates are the 50 input-space nearest neighbors.
    Xc = X - X.min()
    Xc /= Xc.max()
    true = NearestNeighbors(n_neighbors=50).fit(Xc).kneighbors(return_distance=False)
    assert_array_equal(np.sort(cand, axis=1), np.sort(true, axis=1))


def test_rashomon_deterministic():
    X, _ = make_blobs(300, n_features=10, random_state=0)
    a = PaCMAP(n_rashomon=2, random_state=0, **FAST).fit(X)
    b = PaCMAP(n_rashomon=2, random_state=0, **FAST).fit(X)
    assert_array_equal(a.embedding_, b.embedding_)
    assert_array_equal(a.rashomon_scores_, b.rashomon_scores_)


def test_rashomon_off_leaves_output_unchanged():
    X, _ = make_blobs(300, n_features=10, random_state=0)
    est = PaCMAP(random_state=0, **FAST).fit(X)
    assert not hasattr(est, "rashomon_scores_")


def test_rashomon_penalty_changes_selection():
    X, _ = make_blobs(400, n_features=15, centers=5, cluster_std=4.0, random_state=0)
    a = PaCMAP(n_rashomon=3, rashomon_penalty=0.0, random_state=0, **FAST).fit(X)
    b = PaCMAP(n_rashomon=3, rashomon_penalty=10.0, random_state=0, **FAST).fit(X)
    assert not np.array_equal(a.pair_neighbors_, b.pair_neighbors_)


def test_rashomon_purity_on_digits():
    # Refined neighbor graphs should be at least as label-pure as the
    # default scaled-distance graph (the motivation in Sun et al., 2026).
    X, y = load_digits(return_X_y=True)
    purity = lambda P: np.mean(y[P[:, 0]] == y[P[:, 1]])
    base = PaCMAP(random_state=0).fit(X)
    ref = PaCMAP(n_rashomon=5, random_state=0).fit(X)
    assert purity(ref.pair_neighbors_) >= purity(base.pair_neighbors_) - 0.002


@pytest.mark.parametrize("n_rashomon", [1, 0])
def test_rashomon_invalid(n_rashomon):
    with pytest.raises(ValueError, match="n_rashomon"):
        PaCMAP(n_rashomon=n_rashomon).fit(np.random.rand(30, 3))


def test_localmap_has_no_rashomon_parameters():
    # Rashomon refinement fragments LocalMAP embeddings, so LocalMAP does not
    # expose it at all (not settable via __init__, set_params or a search).
    params = LocalMAP().get_params()
    assert "n_rashomon" not in params and "rashomon_penalty" not in params
    with pytest.raises(TypeError):
        LocalMAP(n_rashomon=5)
    with pytest.raises(ValueError, match="Invalid parameter"):
        LocalMAP().set_params(n_rashomon=5)
    X, _ = make_blobs(200, n_features=10, random_state=0)
    assert not hasattr(LocalMAP(random_state=0, **FAST).fit(X), "rashomon_scores_")
