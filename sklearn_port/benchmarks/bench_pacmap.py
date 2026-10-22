"""Compare the scikit-learn port against the reference implementation.

Reports wall time, trustworthiness (on a subsample), 10-NN accuracy and
random-triplet accuracy for both implementations. Run from the repository
root:

    python sklearn_port/benchmarks/bench_pacmap.py [--seeds 3] [--large]

USPS and 20NG come from ``data/`` (plain float arrays). ``--large`` adds a
70000 x 784 synthetic mixture to measure scaling of the exact kNN search
against the reference's HNSW search.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from pacmap_sklearn import PaCMAP

DATA = Path(__file__).resolve().parents[2] / "data"


def load(name):
    X = np.load(DATA / f"{name}.npy", allow_pickle=False).astype(np.float32)
    y = np.load(DATA / f"{name}_labels.npy", allow_pickle=False)
    return X.reshape(len(X), -1), y


def synthetic(n=70000, d=784, k=10, rank=20, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    centers = rng.normal(size=(k, rank)) * 4
    basis = rng.normal(size=(rank, d))
    X = (centers[y] + rng.normal(size=(n, rank))) @ basis
    X += 0.5 * rng.normal(size=(n, d))
    return X.astype(np.float32), y


def triplet_accuracy(X, emb, n_triplets=20000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, len(X), size=(n_triplets, 3))
    dh = lambda a, b: np.linalg.norm(X[t[:, a]] - X[t[:, b]], axis=1)
    dl = lambda a, b: np.linalg.norm(emb[t[:, a]] - emb[t[:, b]], axis=1)
    return np.mean((dh(0, 1) < dh(0, 2)) == (dl(0, 1) < dl(0, 2)))


def scores(X, y, emb, seed=0):
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), size=min(3000, len(X)), replace=False)
    return dict(
        trust=trustworthiness(X[sub], emb[sub], n_neighbors=10),
        knn=cross_val_score(KNeighborsClassifier(10), emb, y, cv=5).mean(),
        triplet=triplet_accuracy(X, emb, seed=seed),
    )


def run(name, X, y, seeds, include_tsne=False):
    import pacmap.pacmap as ref

    ref.PaCMAP(random_state=0).fit_transform(X[:200])  # numba JIT warm-up
    methods = {
        "port": lambda s: PaCMAP(random_state=s),
        "reference": lambda s: ref.PaCMAP(random_state=s, knn_backend="faiss"),
    }
    if include_tsne:
        methods["sklearn TSNE"] = lambda s: TSNE(random_state=s)
    rows = []
    for label, make in methods.items():
        res = []
        for s in range(seeds):
            t0 = time.perf_counter()
            emb = make(s).fit_transform(X)
            res.append(dict(time=time.perf_counter() - t0, **scores(X, y, emb, s)))
        mean = {k: np.mean([r[k] for r in res]) for k in res[0]}
        rows.append((label, mean))
    print(f"\n{name}  (n={X.shape[0]}, d={X.shape[1]}, seeds={seeds})")
    print(f"{'method':<14}{'time[s]':>9}{'trust':>8}{'10-NN':>8}{'triplet':>9}")
    for label, m in rows:
        print(
            f"{label:<14}{m['time']:>9.1f}{m['trust']:>8.4f}"
            f"{m['knn']:>8.4f}{m['triplet']:>9.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    args = parser.parse_args()
    for name in ("USPS", "20NG"):
        run(name, *load(name), args.seeds, args.tsne)
    if args.large:
        run("synthetic-70k", *synthetic(), 1, False)
