import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits, make_blobs
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from pacmap_sklearn import PaCMAP
from pacmap_sklearn._pacmap import _incidence_list, _phase_weights
from pacmap_sklearn._pacmap_utils import pacmap_gradient

FAST = dict(num_iters=(30, 30, 60))


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits(return_X_y=True)
    return X, y


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_bitwise_identical():
    X, _ = make_blobs(300, n_features=20, centers=5, random_state=0)
    a = PaCMAP(random_state=7, **FAST).fit_transform(X)
    b = PaCMAP(random_state=7, **FAST).fit_transform(X)
    assert_array_equal(a, b)


def test_different_seeds_differ():
    X, _ = make_blobs(300, n_features=20, centers=5, random_state=0)
    a = PaCMAP(random_state=0, init="random", **FAST).fit_transform(X)
    b = PaCMAP(random_state=1, init="random", **FAST).fit_transform(X)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("metric", ["euclidean", "angular"])
def test_output_independent_of_thread_count(metric):
    X, _ = make_blobs(500, n_features=150, centers=6, random_state=0)
    with threadpool_limits(limits=1, user_api="openmp"):
        a = PaCMAP(metric=metric, random_state=3, **FAST).fit_transform(X)
    with threadpool_limits(limits=4, user_api="openmp"):
        b = PaCMAP(metric=metric, random_state=3, **FAST).fit_transform(X)
    assert_array_equal(a, b)


def test_instances_do_not_share_random_state():
    # Regression test for the reference implementation, where random_state
    # was stored in a module-level global shared by all instances.
    X, _ = make_blobs(200, n_features=10, random_state=0)
    seeded = PaCMAP(random_state=0, **FAST)
    PaCMAP(random_state=None, **FAST)  # must not reset `seeded`'s state
    a = seeded.fit_transform(X)
    b = PaCMAP(random_state=0, **FAST).fit_transform(X)
    assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def test_pair_invariants():
    X, _ = make_blobs(400, n_features=8, random_state=0)
    est = PaCMAP(random_state=0, num_iters=(0, 0, 0)).fit(X)
    n = X.shape[0]
    nn = est.pair_neighbors_[:, 1].reshape(n, est.n_neighbors_)
    mn = est.pair_mn_[:, 1].reshape(n, est.n_mn_)
    fp = est.pair_fp_[:, 1].reshape(n, est.n_fp_)
    rows = np.arange(n)[:, None]
    for block in (nn, mn, fp):
        assert not np.any(block == rows), "self-pair"
        assert all(len(set(r)) == len(r) for r in block), "duplicate partner"
    for i in range(n):
        assert not set(fp[i]) & set(nn[i]), "FP overlaps NN"
    assert_array_equal(est.pair_neighbors_[:, 0], np.repeat(np.arange(n), est.n_neighbors_))


def test_mn_partner_is_not_nearest_of_candidates():
    # A mid-near partner is the 2nd closest of 6 random candidates, so on
    # average it should be much farther than the nearest neighbors and much
    # closer than further pairs.
    X, _ = make_blobs(1000, n_features=5, centers=10, random_state=0)
    est = PaCMAP(random_state=0, num_iters=(0, 0, 0)).fit(X)
    d = lambda P: np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1).mean()
    assert d(est.pair_neighbors_) < d(est.pair_mn_) < d(est.pair_fp_)


def test_precomputed_matches_euclidean():
    # Global min-max scaling and centering leave the locally scaled distances
    # and the mid-near ranking unchanged, so with the same seed and a random
    # init the two code paths must sample identical pairs.
    X = np.random.RandomState(0).rand(250, 6)
    D = cdist(X, X)
    kw = dict(init="random", random_state=0, **FAST)
    a = PaCMAP(**kw).fit(X)
    b = PaCMAP(metric="precomputed", **kw).fit(D)
    assert_array_equal(a.pair_neighbors_, b.pair_neighbors_)
    assert_array_equal(a.pair_mn_, b.pair_mn_)
    assert_array_equal(a.pair_fp_, b.pair_fp_)
    assert_allclose(a.embedding_, b.embedding_, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "n_samples, expected",
    [(5000, 10), (10000, 10), (100000, 25), (1000000, 40)],
)
def test_auto_n_neighbors(n_samples, expected):
    assert PaCMAP(n_neighbors=None)._n_pairs(n_samples)[0] == expected


@pytest.mark.parametrize("n_samples", [3, 4, 5, 8, 12, 30])
def test_tiny_datasets(n_samples):
    X = np.random.RandomState(0).rand(n_samples, 4)
    emb = PaCMAP(random_state=0, **FAST).fit_transform(X)
    assert emb.shape == (n_samples, 2)
    assert np.all(np.isfinite(emb))


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def test_phase_schedule():
    assert _phase_weights(0, 100, 100) == (2.0, 1000.0, 1.0)
    assert _phase_weights(50, 100, 100) == pytest.approx((2.0, 501.5, 1.0))
    assert _phase_weights(150, 100, 100) == (3.0, 3.0, 1.0)
    assert _phase_weights(200, 100, 100) == (1.0, 0.0, 1.0)


def test_gradient_matches_finite_differences():
    rng = np.random.RandomState(0)
    n = 30
    Y = rng.randn(n, 2) * 2
    pairs = np.column_stack([np.repeat(np.arange(n), 6), rng.randint(0, n, 6 * n)])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]].astype(np.intp)
    types = rng.randint(0, 3, len(pairs)).astype(np.int8)
    indptr, other, t = _incidence_list(pairs, types, n)
    w = (2.0, 7.0, 1.5)

    def loss(Y):
        g, nl = np.empty_like(Y), np.empty(n)
        pacmap_gradient(np.ascontiguousarray(Y), indptr, other, t, *w, g, nl, 1)
        return 0.5 * nl.sum(), g

    _, grad = loss(Y)
    eps = 1e-6
    num = np.empty_like(Y)
    for i in range(n):
        for k in range(2):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[i, k] += eps
            Ym[i, k] -= eps
            num[i, k] = (loss(Yp)[0] - loss(Ym)[0]) / (2 * eps)
    assert_allclose(grad, num, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Embedding quality
# ---------------------------------------------------------------------------


def test_digits_quality(digits):
    X, y = digits
    emb = PaCMAP(random_state=0).fit_transform(X)
    assert trustworthiness(X, emb, n_neighbors=10) > 0.97
    knn_acc = cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean()
    assert knn_acc > 0.95


@pytest.mark.parametrize("metric", ["manhattan", "angular"])
def test_other_metrics_quality(digits, metric):
    X, y = digits
    emb = PaCMAP(metric=metric, random_state=0).fit_transform(X)
    knn_acc = cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean()
    assert knn_acc > 0.9


def test_hamming_metric():
    rng = np.random.RandomState(0)
    centers = rng.rand(4, 64) < 0.5
    y = rng.randint(0, 4, 400)
    X = np.logical_xor(centers[y], rng.rand(400, 64) < 0.1).astype(float)
    emb = PaCMAP(metric="hamming", random_state=0, **FAST).fit_transform(X)
    knn_acc = cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean()
    assert knn_acc > 0.95


def test_pca_preprocessing_path():
    X, y = make_blobs(600, n_features=300, centers=5, random_state=0)
    emb = PaCMAP(random_state=0, **FAST).fit_transform(X)
    knn_acc = cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean()
    assert knn_acc > 0.99


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_num_iters_int():
    X = np.random.RandomState(0).rand(50, 3)
    assert PaCMAP(num_iters=7, random_state=0).fit(X).n_iter_ == 207


def test_array_init():
    X = np.random.RandomState(0).rand(50, 3)
    init = np.random.RandomState(1).rand(50, 2)
    emb = PaCMAP(init=init, random_state=0, **FAST).fit_transform(X)
    assert emb.shape == (50, 2)


def test_feature_names_out():
    X = np.random.RandomState(0).rand(50, 3)
    est = PaCMAP(n_components=3, random_state=0, **FAST).fit(X)
    assert_array_equal(est.get_feature_names_out(), ["pacmap0", "pacmap1", "pacmap2"])


@pytest.mark.parametrize(
    "params, X, match",
    [
        (dict(num_iters=(1, 2)), np.ones((20, 3)), "num_iters"),
        (dict(num_iters=(1, 2, -1)), np.ones((20, 3)), "num_iters"),
        (dict(init=np.ones((5, 2))), np.random.rand(20, 3), "init must have shape"),
        (dict(metric="precomputed"), np.random.rand(20, 20), 'init="pca"'),
        (dict(metric="precomputed", init="random"), np.random.rand(20, 5), "square"),
        (dict(metric="precomputed", init="random"), -np.ones((20, 20)), "Negative"),
        (dict(fp_ratio=0.01), np.random.rand(20, 3), "further pair"),
    ],
)
def test_invalid_input(params, X, match):
    with pytest.raises(ValueError, match=match):
        PaCMAP(**params).fit(X)
