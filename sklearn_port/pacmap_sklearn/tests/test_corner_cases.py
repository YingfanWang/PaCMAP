"""Corner cases, each tied to a failure mode scikit-learn's TSNE has had.

Issue numbers refer to github.com/scikit-learn/scikit-learn.
"""

import time
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.spatial.distance import cdist
from sklearn import config_context
from sklearn.datasets import make_blobs
from threadpoolctl import threadpool_limits

from functools import partial

from pacmap_sklearn import PaCMAP

LocalMAP = partial(PaCMAP, localmap=True)  # LocalMAP is a switch on PaCMAP
from pacmap_sklearn._pacmap_utils import resample_fp_nearby, sample_fp_pairs

FAST = dict(num_iters=(20, 20, 20), random_state=0)
ESTIMATORS = [PaCMAP, LocalMAP]


# --- constant / duplicate data (#28368, #32788: NaN then crash) -------------


@pytest.mark.parametrize("Est", ESTIMATORS)
@pytest.mark.parametrize(
    "X",
    [np.ones((10, 5)), np.zeros((91, 3)), np.ones((10, 200))],
    ids=["ones-10x5", "zeros-91x3", "ones-10x200-pca-path"],
)
def test_identical_samples_warn_and_stay_finite(Est, X):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no numpy noise
        with pytest.warns(UserWarning, match="All samples are identical"):
            Y = Est(**FAST).fit_transform(X)
    assert np.all(np.isfinite(Y))


@pytest.mark.parametrize("Est", ESTIMATORS)
def test_heavy_duplicates(Est):
    # >= 6 copies of every point: sigma_i (mean distance to neighbors 4-6)
    # is 0, the analogue of t-SNE's zero-bandwidth rows.
    X = np.repeat(np.random.RandomState(0).rand(20, 5), 30, axis=0)
    Y = Est(**FAST).fit_transform(X)
    assert np.all(np.isfinite(Y))
    assert np.ptp(Y, axis=0).min() > 1


# --- small n relative to requested pairs (#10805, #23471) --------------------


def test_small_n_warns_about_reduced_pairs():
    X = np.random.RandomState(0).rand(12, 3)
    with pytest.warns(UserWarning, match="too small for the requested pairs"):
        PaCMAP(**FAST).fit(X)


def test_no_warning_when_pairs_fit():
    X = np.random.RandomState(0).rand(200, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PaCMAP(**FAST).fit(X)


# --- n_components extremes (#4154, #9941) ------------------------------------


@pytest.mark.parametrize("n_components", [1, 3, 10, 25])
def test_n_components_range(n_components):
    X = np.random.RandomState(0).rand(150, 30)
    Y = PaCMAP(n_components=n_components, **FAST).fit_transform(X)
    assert Y.shape == (150, n_components) and np.all(np.isfinite(Y))


# --- dtypes and output (#4124, #19471 underflow, float32 internals) ----------


@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
def test_extreme_scales_give_same_embedding(scale):
    # Min-max scaling makes the result scale-invariant; nothing may
    # underflow or overflow on the way.
    X = np.random.RandomState(0).rand(200, 5)
    ref = PaCMAP(**FAST).fit_transform(X)
    Y = PaCMAP(**FAST).fit_transform(X * scale)
    np.testing.assert_allclose(Y, ref, rtol=1e-6, atol=1e-6)


def test_float32_and_float64_input_agree():
    X = np.random.RandomState(0).rand(200, 5).astype(np.float32)
    a = PaCMAP(**FAST).fit_transform(X)
    b = PaCMAP(**FAST).fit_transform(X.astype(np.float64))
    assert a.dtype == b.dtype == np.float64
    assert_array_equal(a, b)


# --- set_output leaking into internal estimators (#25365) --------------------


@pytest.mark.parametrize("n_features", [5, 150], ids=["pca-init", "svd-path"])
@pytest.mark.parametrize("Est", ESTIMATORS)
def test_pandas_transform_output(Est, n_features):
    pd = pytest.importorskip("pandas")
    X = np.random.RandomState(0).rand(200, n_features)
    expected = Est(**FAST).fit_transform(X)
    with config_context(transform_output="pandas"):
        Y = Est(**FAST).fit_transform(X)
    assert isinstance(Y, pd.DataFrame)
    assert list(Y.columns) == ["pacmap0", "pacmap1"]
    assert_array_equal(Y.to_numpy(), expected)


def test_pandas_output_with_array_init():
    pytest.importorskip("pandas")
    X = np.random.RandomState(0).rand(100, 5)
    init = np.random.RandomState(1).rand(100, 2)
    with config_context(transform_output="pandas"):
        PaCMAP(init=init, **FAST).fit_transform(X)


# --- init validation (#7204, plus NaN init found by probing) -----------------


@pytest.mark.parametrize(
    "init, match",
    [
        (np.full((50, 2), np.nan), "init"),
        (np.full((50, 2), np.inf), "init"),
        (np.ones((50, 3)), "init must have shape"),
        (np.ones(50), "2D array"),
    ],
)
def test_invalid_init_array(init, match):
    with pytest.raises(ValueError, match=match):
        PaCMAP(init=init).fit(np.random.RandomState(0).rand(50, 4))


def test_constant_init_column_warns():
    X = np.random.RandomState(0).rand(50, 4)
    init = np.c_[np.random.RandomState(1).rand(50), np.ones(50)]
    with pytest.warns(UserWarning, match="constant column"):
        PaCMAP(init=init, **FAST).fit(X)


# --- precomputed numerics (#11376, #5772, #23428) ----------------------------


def test_precomputed_huge_distances_no_overflow():
    rng = np.random.RandomState(0)
    P = rng.rand(100, 3)
    D = cdist(P, P)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        a = PaCMAP(metric="precomputed", init="random", **FAST).fit_transform(D)
        b = PaCMAP(metric="precomputed", init="random", **FAST).fit_transform(D * 1e200)
    assert_array_equal(a, b)  # the scaled re-ranking is scale-invariant


def test_precomputed_negative_values_rejected():
    D = -np.ones((30, 30))
    with pytest.raises(ValueError, match="Negative values"):
        PaCMAP(metric="precomputed", init="random").fit(D)


# --- reported loss: never stale or NaN (#6507, #10593) -----------------------


@pytest.mark.parametrize("Est", ESTIMATORS)
@pytest.mark.parametrize("num_iters", [(0, 0, 0), (0, 0, 1), (1, 0, 0), (3, 7, 13)])
def test_loss_is_final_and_finite(Est, num_iters):
    X = np.random.RandomState(0).rand(100, 5)
    est = Est(num_iters=num_iters, random_state=0).fit(X)
    assert np.isfinite(est.loss_)
    assert est.n_iter_ == sum(num_iters)


def test_loss_matches_final_embedding():
    from pacmap_sklearn._pacmap import _incidence_list, _pair_types, _phase_weights
    from pacmap_sklearn._pacmap_utils import pacmap_gradient

    X = np.random.RandomState(0).rand(100, 5)
    est = PaCMAP(num_iters=(5, 5, 5), random_state=0).fit(X)
    P = [est.pair_neighbors_, est.pair_mn_, est.pair_fp_]
    ip, ot, ty = _incidence_list(np.concatenate(P), _pair_types(*map(len, P)), 100)
    g, nl = np.empty_like(est.embedding_), np.empty(100)
    pacmap_gradient(est.embedding_, ip, ot, ty, *_phase_weights(14, 5, 5), g, nl, 1)
    assert est.loss_ == pytest.approx(0.5 * nl.sum())


# --- divergence: no silent NaN embeddings (#6665, #9026) ---------------------


def test_divergence_raises_clear_error(monkeypatch):
    import pacmap_sklearn._pacmap as mod

    def explode(self, Y, grad, itr):
        Y[:] = np.inf

    monkeypatch.setattr(mod._Adam, "update", explode)
    with pytest.raises(ValueError, match="diverged"):
        PaCMAP(**FAST).fit(np.random.RandomState(0).rand(50, 3))


def test_huge_learning_rate_stays_finite():
    Y = PaCMAP(learning_rate=1e6, **FAST).fit_transform(
        np.random.RandomState(0).rand(200, 5)
    )
    assert np.all(np.isfinite(Y))


# --- threads: no hidden nondeterminism, all-zero input with n_jobs (#16184) --


@pytest.mark.parametrize("Est", ESTIMATORS)
def test_ties_are_thread_independent(Est):
    # Regular grid / integer data: many tied distances (#15821).
    X = np.random.RandomState(0).randint(0, 3, size=(400, 4)).astype(float)
    with threadpool_limits(1, user_api="openmp"):
        a = Est(**FAST).fit_transform(X)
    with threadpool_limits(4, user_api="openmp"):
        b = Est(**FAST).fit_transform(X)
    assert_array_equal(a, b)


def test_n_jobs_on_all_zero_input():
    with pytest.warns(UserWarning, match="identical"):
        PaCMAP(n_jobs=2, **FAST).fit(np.zeros((100, 512)))


# --- memory: Rashomon candidate distances are chunked (#7089, #18966) --------


def test_rashomon_chunking_matches_unchunked(monkeypatch):
    import pacmap_sklearn._pacmap as mod

    X, _ = make_blobs(300, n_features=8, random_state=0)
    a = PaCMAP(n_rashomon=2, **FAST).fit(X)
    monkeypatch.setattr(mod, "_RASHOMON_CHUNK", 7)
    b = PaCMAP(n_rashomon=2, **FAST).fit(X)
    assert_array_equal(a.rashomon_scores_, b.rashomon_scores_)


# --- LocalMAP: bounded resampling for isolated points -------------------------


def test_localmap_resampling_bounded_for_isolated_points():
    rng = np.random.RandomState(0)
    n = 20000
    Y = rng.rand(n, 2) * 1e4  # nobody within the threshold of anybody
    nbrs = np.ascontiguousarray((np.arange(n)[:, None] + np.arange(1, 11)) % n,
                                dtype=np.intp)
    old = sample_fp_pairs(nbrs, 20, 1, 1)
    t0 = time.perf_counter()
    new = resample_fp_nearby(Y, nbrs, old, 1e-3, 0, 0, 1)
    assert time.perf_counter() - t0 < 5  # reference behavior: ~6 s on 2 threads
    assert_array_equal(new, old)  # every slot falls back to its old partner


def test_localmap_tiny_threshold_completes():
    X = np.random.RandomState(0).rand(3000, 5)
    t0 = time.perf_counter()
    LocalMAP(low_dist_thres=1e-6, **FAST).fit(X)
    assert time.perf_counter() - t0 < 30


# --- input validation basics (#4615) ------------------------------------------


@pytest.mark.parametrize("shape", [(5, 0), (1, 3), (2, 3)])
def test_degenerate_shapes_rejected(shape):
    with pytest.raises(ValueError):
        PaCMAP().fit(np.ones(shape))


# --- runtime warnings for unverified territory --------------------------------


def test_rashomon_warns_experimental():
    X = np.random.RandomState(0).rand(200, 5)
    with pytest.warns(UserWarning, match="n_rashomon is experimental"):
        PaCMAP(n_rashomon=2, **FAST).fit(X)


def test_no_rashomon_warning_by_default():
    X = np.random.RandomState(0).rand(200, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PaCMAP(**FAST).fit(X)
        LocalMAP(**FAST).fit(X)


def test_localmap_warns_when_threshold_too_small():
    X = np.random.RandomState(0).rand(500, 5)
    with pytest.warns(UserWarning, match="local graph adjustment is largely inactive"):
        LocalMAP(low_dist_thres=1e-6, **FAST).fit(X)


def test_large_n_exact_search_warns(monkeypatch):
    import pacmap_sklearn._pacmap as mod

    monkeypatch.setattr(mod, "_EXACT_KNN_WARN_N", 100)
    X = np.random.RandomState(0).rand(200, 5)
    with pytest.warns(UserWarning, match="Exact nearest-neighbor search"):
        PaCMAP(**FAST).fit(X)
    with warnings.catch_warnings():  # not with an ANN transformer
        warnings.simplefilter("error")
        from sklearn.neighbors import KNeighborsTransformer

        PaCMAP(neighbors_transformer=KNeighborsTransformer(n_neighbors=61, mode="distance"),
               **FAST).fit(X)


def test_openmp_disabled_warns(monkeypatch):
    import pacmap_sklearn._pacmap as mod

    monkeypatch.setattr(mod, "OPENMP_ENABLED", False)
    with pytest.warns(UserWarning, match="compiled without OpenMP"):
        PaCMAP(**FAST).fit(np.random.RandomState(0).rand(100, 5))


def test_this_build_has_openmp():
    from pacmap_sklearn._pacmap_utils import OPENMP_ENABLED

    assert OPENMP_ENABLED
