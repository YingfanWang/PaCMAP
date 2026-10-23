import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer

from functools import partial

from pacmap_sklearn import PaCMAP

LocalMAP = partial(PaCMAP, localmap=True)  # LocalMAP is a switch on PaCMAP

FAST = dict(num_iters=(30, 30, 60))


@pytest.mark.parametrize("Est", [PaCMAP, LocalMAP])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_exact_transformer_reproduces_internal_search(Est, metric):
    X, _ = make_blobs(400, n_features=20, centers=5, random_state=0)
    kw = dict(metric=metric, random_state=0, **FAST)
    a = Est(**kw).fit(X)
    b = Est(
        neighbors_transformer=KNeighborsTransformer(
            n_neighbors=61, mode="distance", metric=metric
        ),
        **kw,
    ).fit(X)
    assert_array_equal(a.pair_neighbors_, b.pair_neighbors_)
    assert_array_equal(a.embedding_, b.embedding_)
    assert hasattr(b, "neighbors_transformer_")
    assert not hasattr(a, "neighbors_transformer_")


def test_transformer_fitted_on_preprocessed_data():
    # With > 100 features the transformer must see the 100-dim projection.
    class Spy(KNeighborsTransformer):
        def fit_transform(self, X, y=None):
            self.seen_shape_ = X.shape
            return super().fit_transform(X, y)

    X, _ = make_blobs(300, n_features=150, random_state=0)
    est = PaCMAP(
        neighbors_transformer=Spy(n_neighbors=61, mode="distance"),
        random_state=0, **FAST,
    ).fit(X)
    assert est.neighbors_transformer_.seen_shape_ == (300, 100)


def test_user_transformer_is_not_mutated():
    t = KNeighborsTransformer(n_neighbors=61, mode="distance")
    PaCMAP(neighbors_transformer=t, random_state=0, **FAST).fit(
        np.random.RandomState(0).rand(100, 4)
    )
    assert not hasattr(t, "n_samples_fit_")


def test_rashomon_with_transformer():
    X, _ = make_blobs(300, n_features=10, random_state=0)
    t = KNeighborsTransformer(n_neighbors=61, mode="distance")
    a = PaCMAP(n_rashomon=2, random_state=0, **FAST).fit(X)
    b = PaCMAP(n_rashomon=2, neighbors_transformer=t, random_state=0, **FAST).fit(X)
    assert_array_equal(a.pair_neighbors_, b.pair_neighbors_)


class _SelfFreeKNN(TransformerMixin, BaseEstimator):
    """Graph without self-loops and with rows in scrambled order."""

    def __init__(self, n_neighbors=60):
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        g = KNeighborsTransformer(n_neighbors=self.n_neighbors, mode="distance")
        g = g.fit(X).kneighbors_graph(None, mode="distance").tocsr()
        rng = np.random.RandomState(0)
        for i in range(g.shape[0]):
            sl = slice(g.indptr[i], g.indptr[i + 1])
            perm = rng.permutation(sl.stop - sl.start)
            g.indices[sl] = g.indices[sl][perm]
            g.data[sl] = g.data[sl][perm]
        return csr_array(g)


def test_graph_without_self_loops_and_unsorted_rows():
    X, _ = make_blobs(300, n_features=10, random_state=0)
    a = PaCMAP(random_state=0, **FAST).fit(X)
    b = PaCMAP(neighbors_transformer=_SelfFreeKNN(60), random_state=0, **FAST).fit(X)
    assert_array_equal(a.pair_neighbors_, b.pair_neighbors_)


def test_small_graph_warns_and_too_small_raises():
    X = np.random.RandomState(0).rand(200, 5)
    with pytest.warns(UserWarning, match="candidates for the scaled re-ranking"):
        PaCMAP(neighbors_transformer=KNeighborsTransformer(n_neighbors=20, mode="distance"),
               random_state=0, **FAST).fit(X)
    with pytest.raises(ValueError, match="at least n_neighbors=10"):
        PaCMAP(neighbors_transformer=KNeighborsTransformer(n_neighbors=5, mode="distance"),
               **FAST).fit(X)


@pytest.mark.parametrize(
    "bad, match",
    [
        (lambda X: np.zeros((len(X), len(X))), "sparse distance graph"),
        (lambda X: csr_array((len(X), len(X) + 1)), "sparse distance graph"),
        (lambda X: -KNeighborsTransformer(n_neighbors=61, mode="distance").fit_transform(X),
         "negative"),
    ],
)
def test_invalid_graphs(bad, match):
    class Bad(TransformerMixin, BaseEstimator):
        def fit_transform(self, X, y=None):
            return bad(X)

    with pytest.raises(ValueError, match=match):
        PaCMAP(neighbors_transformer=Bad(), **FAST).fit(np.random.rand(100, 5))


def test_precomputed_rejects_transformer():
    D = np.random.rand(50, 50)
    D = D + D.T
    np.fill_diagonal(D, 0)
    with pytest.raises(ValueError, match="cannot be used with metric"):
        PaCMAP(metric="precomputed", init="random",
               neighbors_transformer=KNeighborsTransformer()).fit(D)


def test_approximate_transformer_quality():
    pynndescent = pytest.importorskip("pynndescent")
    X, y = load_digits(return_X_y=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = pynndescent.PyNNDescentTransformer(n_neighbors=61, random_state=0)
        emb = PaCMAP(neighbors_transformer=t, random_state=0).fit_transform(X)
    assert cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean() > 0.95
