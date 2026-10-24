"""Speed of the original PaCMAP (numba + faiss HNSW) vs the scikit-learn
implementation (Cython + exact kNN, and with a faiss-HNSW
neighbors_transformer for an apples-to-apples neighbor search).

    python sklearn_port/benchmarks/bench_impl_speed.py OUT.json

Phases are timed separately for both implementations: preprocessing (PCA),
nearest-neighbor search, and the rest (pair sampling + optimization).
"""

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
import pacmap.pacmap as ref  # original implementation
from bench_ann import FaissHNSWTransformer
from bench_vs_tsne import _npy, _npy_zip
from sklearn.datasets import load_digits
from sklearn.manifold import PaCMAP
from sklearn.manifold import _pacmap as skmod


def _timed(store, key, fn):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0
        return out

    return wrapper


def _reset_faiss_threads():
    # The original sets faiss.omp_set_num_threads(1) globally whenever
    # random_state is given; undo it so every run starts from the default.
    import faiss

    faiss.omp_set_num_threads(os.cpu_count())


def time_reference(X, seed):
    _reset_faiss_threads()
    T = {}
    orig = (ref.preprocess_X, ref.generate_pair, ref.pacmap)
    ref.preprocess_X = _timed(T, "preprocess", orig[0])
    ref.generate_pair = _timed(T, "pairs (incl. kNN)", orig[1])
    ref.pacmap = _timed(T, "optimize", orig[2])
    knn_orig = ref.compute_nearest_neighbors
    ref.compute_nearest_neighbors = _timed(T, "kNN", knn_orig)
    try:
        t0 = time.perf_counter()
        ref.PaCMAP(random_state=seed).fit_transform(X)
        T["total"] = time.perf_counter() - t0
    finally:
        ref.preprocess_X, ref.generate_pair, ref.pacmap = orig
        ref.compute_nearest_neighbors = knn_orig
    return T


def time_sklearn(X, seed, transformer=None):
    _reset_faiss_threads()
    T = {}
    cls = PaCMAP
    saved = {m: cls.__dict__[m] for m in ("_preprocess", "_knn", "_optimize")}
    cls._preprocess = _timed(T, "preprocess", saved["_preprocess"])
    cls._knn = _timed(T, "kNN", saved["_knn"])
    cls._optimize = _timed(T, "optimize", saved["_optimize"])
    try:
        t0 = time.perf_counter()
        PaCMAP(random_state=seed, neighbors_transformer=transformer).fit_transform(X)
        T["total"] = time.perf_counter() - t0
    finally:
        for m, f in saved.items():
            setattr(cls, m, f)
    return T


def datasets():
    yield "Digits (1.8k×64)", load_digits().data
    yield "USPS (9.3k×256)", _npy("USPS.npy").reshape(9298, -1)
    yield "20NG (18.8k×100)", _npy("20NG.npy")
    mnist = _npy_zip("mnist_images.npy.zip").reshape(70000, -1)
    yield "MNIST 20k (×784)", mnist[np.random.RandomState(0).choice(70000, 20000, False)]
    yield "MNIST 70k (×784)", mnist
    fmnist = _npy_zip("fmnist_images.npy.zip").reshape(70000, -1)
    yield "F-MNIST 70k (×784)", fmnist


def main(out_path, n_seeds=2):
    warnings.simplefilter("ignore")
    # JIT warm-up for numba so compilation is not counted.
    ref.PaCMAP(random_state=0).fit_transform(np.random.rand(500, 20).astype(np.float32))
    rows = []
    for name, X in datasets():
        X = np.ascontiguousarray(X, dtype=np.float32)
        for label, fn in [
            ("original, seeded (reproducible)", time_reference),
            ("original, unseeded (fastest mode)", lambda X, s: time_reference(X, None)),
            ("scikit-learn (exact kNN)", time_sklearn),
            (
                "scikit-learn + faiss HNSW transformer",
                lambda X, s: time_sklearn(X, s, FaissHNSWTransformer(n_neighbors=61)),
            ),
        ]:
            for s in range(n_seeds):
                T = fn(X, s)
                rows.append(dict(dataset=name, impl=label, seed=s, **T))
                print(json.dumps(rows[-1]), flush=True)
                json.dump(rows, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
