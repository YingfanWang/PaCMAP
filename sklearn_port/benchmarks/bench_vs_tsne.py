"""PaCMAP vs t-SNE (PCA init, the current default; random init, the old
default used in the PaCMAP paper) across datasets with different structure.

    python sklearn_port/benchmarks/bench_vs_tsne.py OUT.json [--seeds 2]

Uses sklearn.manifold.PaCMAP if available (scikit-learn fork), else the
staging package. Run from the repository root (reads data/).
"""

import io
import json
import pickle
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.datasets import load_digits, make_s_curve
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

try:
    from sklearn.manifold import PaCMAP
except ImportError:
    from pacmap_sklearn import PaCMAP

DATA = Path(__file__).resolve().parents[2] / "data"


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickle numpy arrays only; refuse any other global (no code exec)."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        # Python 2-era pickles store byte strings via _codecs.encode(str,
        # "latin1"): a pure string -> bytes conversion, no code execution.
        ("_codecs", "encode"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing global {module}.{name}")
        return super().find_class(module, name)


def _load_npy(raw):
    """Load a .npy payload; object arrays go through the restricted unpickler."""
    f = io.BytesIO(raw)
    if not raw.startswith(b"\x93NUMPY"):
        # A raw (Python 2-era) pickle of an ndarray, not an .npy file.
        arr = np.asarray(_NumpyOnlyUnpickler(f, encoding="latin1").load())
        if arr.dtype == object:
            arr = np.stack([np.asarray(a, dtype=np.float64).ravel() for a in arr])
        return arr
    version = np.lib.format.read_magic(f)
    read_header = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    shape, fortran, dtype = read_header(f)
    if dtype.hasobject:
        arr = _NumpyOnlyUnpickler(f).load()
    else:
        f.seek(0)
        arr = np.load(f, allow_pickle=False)
    arr = np.asarray(arr)
    if arr.dtype == object:
        arr = np.stack([np.asarray(a, dtype=np.float64).ravel() for a in arr])
    return arr


def _npy(name):
    return _load_npy((DATA / name).read_bytes())


def _npy_zip(name):
    with zipfile.ZipFile(DATA / name) as z:
        inner = [n for n in z.namelist() if n.endswith(".npy")][0]
        return _load_npy(z.read(inner))


def hierarchical(seed=0):
    rng = np.random.RandomState(seed)
    macro = rng.randn(5, 50) * np.linspace(6, 30, 5)[:, None]
    X, y = [], []
    for m in range(5):
        for s in range(4):
            c = macro[m] + 2.5 * rng.randn(50)
            X.append(c + 0.6 * rng.randn(100, 50))
            y += [m * 4 + s] * 100
    return np.vstack(X), np.array(y)


def datasets(subsample=20000):
    rng = np.random.RandomState(0)

    def sub(X, y):
        if len(X) > subsample:
            i = rng.choice(len(X), subsample, replace=False)
            return X[i], (None if y is None else y[i])
        return X, y

    mammoth = np.asarray(json.loads((DATA / "mammoth_3d.json").read_text()), float)
    S, s_t = make_s_curve(3000, random_state=0)
    yield "Mammoth (10k×3)", mammoth, None
    yield "S-curve (3k×3)", S, None
    yield "Clusters of clusters (2k×50)", *hierarchical()
    yield "Digits (1.8k×64)", *load_digits(return_X_y=True)
    coil = _npy("coil_20.npy").reshape(1440, -1)
    yield "COIL-20 (1.4k×16k)", coil, _npy("coil_20_labels.npy").astype(int)
    yield "USPS (9.3k×256)", _npy("USPS.npy").reshape(9298, -1), _npy("USPS_labels.npy")
    m = _npy_zip("mnist_images.npy.zip").reshape(70000, -1)
    yield "MNIST (20k of 70k×784)", *sub(m, _npy("mnist_labels.npy").astype(int))
    f = _npy_zip("fmnist_images.npy.zip").reshape(70000, -1)
    yield "F-MNIST (20k of 70k×784)", *sub(f, _npy("fmnist_labels.npy").astype(int))
    yield "20NG (18.8k×100)", _npy("20NG.npy"), _npy("20NG_labels.npy")


def scores(X, y, E, seed=0):
    rng = np.random.RandomState(seed)
    n = len(X)
    s3k = rng.choice(n, min(3000, n), replace=False)
    s1k = rng.choice(n, min(1000, n), replace=False)
    t = rng.randint(0, n, size=(20000, 3))

    def closer(Z):
        return np.linalg.norm(Z[t[:, 0]] - Z[t[:, 1]], axis=1) < np.linalg.norm(
            Z[t[:, 0]] - Z[t[:, 2]], axis=1
        )

    out = dict(
        trust=trustworthiness(X[s3k], E[s3k], n_neighbors=10),
        triplet=float(np.mean(closer(X) == closer(E))),
        dist_corr=spearmanr(pdist(X[s1k]), pdist(E[s1k])).statistic,
    )
    if y is not None:
        out["knn"] = cross_val_score(KNeighborsClassifier(10), E, y, cv=5).mean()
        g = np.unique(y)
        hi = np.array([X[y == c].mean(0) for c in g])
        lo = np.array([E[y == c].mean(0) for c in g])
        out["centroid_corr"] = spearmanr(pdist(hi), pdist(lo)).statistic
    return out


METHODS = {
    "PaCMAP": lambda s: PaCMAP(random_state=s),
    "t-SNE (PCA init)": lambda s: TSNE(random_state=s),
    "t-SNE (random init)": lambda s: TSNE(init="random", random_state=s),
}


def main(out_path, n_seeds):
    results = []
    emb_path = str(Path(out_path).with_suffix(".npz"))
    saved = {}
    for name, X, y in datasets():
        X = np.asarray(X, dtype=np.float32)
        for method, make in METHODS.items():
            for s in range(n_seeds):
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    E = make(s).fit_transform(X)
                dt = time.perf_counter() - t0
                r = dict(dataset=name, method=method, seed=s, time=dt,
                         **scores(X, y, E, s))
                results.append(r)
                if s == 0:
                    saved[f"{name}|{method}"] = E.astype(np.float32)
                    if y is not None:
                        saved[f"{name}|labels"] = np.asarray(y)
                    np.savez_compressed(emb_path, **saved)
                print(json.dumps(r), flush=True)
                Path(out_path).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 2
    main(sys.argv[1], seeds)
