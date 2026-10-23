"""PaCMAP with approximate nearest neighbors via ``neighbors_transformer``.

Follows the pattern of scikit-learn's gallery example
"Approximate nearest neighbors in TSNE": a KNeighborsTransformer-style
wrapper around an ANN library, plugged into the estimator.

    PYTHONPATH=sklearn_port:sklearn_port/benchmarks \
        python sklearn_port/benchmarks/bench_ann.py [--n 70000]

Requires ``pynndescent`` and ``faiss-cpu`` (benchmark-only dependencies).
"""

import argparse
import time
import warnings

import numpy as np
from bench_pacmap import scores, synthetic
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsTransformer

from pacmap_sklearn import PaCMAP


class FaissHNSWTransformer(TransformerMixin, BaseEstimator):
    """Euclidean HNSW kNN graph (KNeighborsTransformer convention), using
    the same index settings as the reference implementation (M=32)."""

    def __init__(self, n_neighbors=61, M=32):
        self.n_neighbors = n_neighbors
        self.M = M

    def fit(self, X, y=None):
        import faiss

        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index_ = faiss.IndexHNSWFlat(X.shape[1], self.M, faiss.METRIC_L2)
        self.index_.add(X)
        self.n_samples_fit_ = X.shape[0]
        return self

    def transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        k = self.n_neighbors + 1  # + self, as KNeighborsTransformer does
        D, I = self.index_.search(X, k)
        n = X.shape[0]
        return csr_array(
            (np.sqrt(np.maximum(D, 0)).ravel(), I.ravel(), np.arange(0, n * k + 1, k)),
            shape=(n, self.n_samples_fit_),
        )

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def nn_recall(est, exact):
    """Fraction of the exact run's NN pairs also chosen by ``est``."""
    a = set(map(tuple, exact.pair_neighbors_))
    return np.mean([tuple(p) in a for p in est.pair_neighbors_])


def main(n):
    import pynndescent

    X, y = synthetic(n=n)
    # JIT warm-up so pynndescent's timing excludes numba compilation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pynndescent.PyNNDescentTransformer(n_neighbors=10).fit_transform(X[:2000, :100])

    k = 10 + 50 + 1  # n_neighbors + 50 candidates + self
    runs = {
        "exact (default)": None,
        "KNeighborsTransformer": KNeighborsTransformer(n_neighbors=k, mode="distance"),
        "PyNNDescentTransformer": pynndescent.PyNNDescentTransformer(
            n_neighbors=k, random_state=0, n_jobs=-1
        ),
        "faiss HNSW (gallery-style)": FaissHNSWTransformer(n_neighbors=k),
    }
    exact = None
    print(f"synthetic n={n}, d=784 (searched in the 100-dim PCA projection)")
    print(f"{'neighbors':<28}{'fit[s]':>8}{'knn[s]':>8}{'NN recall':>10}"
          f"{'trust':>8}{'10-NN':>8}{'triplet':>9}")
    for label, t in runs.items():
        est = PaCMAP(random_state=0, neighbors_transformer=t)
        knn_time = []
        orig = est._knn

        def timed_knn(*a, _orig=orig, **kw):
            t0 = time.perf_counter()
            out = _orig(*a, **kw)
            knn_time.append(time.perf_counter() - t0)
            return out

        est._knn = timed_knn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            est.fit(X)
            total = time.perf_counter() - t0
        if exact is None:
            exact = est
        s = scores(X, y, est.embedding_, 0)
        print(f"{label:<28}{total:>8.1f}{knn_time[0]:>8.1f}"
              f"{nn_recall(est, exact):>10.4f}{s['trust']:>8.4f}"
              f"{s['knn']:>8.4f}{s['triplet']:>9.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=70000)
    main(parser.parse_args().n)
