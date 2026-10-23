"""Run edge-case inputs through PaCMAP / LocalMAP, each in a subprocess with
a timeout, and report crash / hang / NaN / non-determinism.

    PYTHONPATH=sklearn_port python sklearn_port/benchmarks/probe_edge_cases.py

Each case is a small Python snippet that defines ``X`` (and optionally
``kw``, extra estimator kwargs). Outcomes:

  ok        finite embedding of the right shape, identical across 2 runs
  ERROR     raised an exception (message shown; may be the intended error)
  NAN       non-finite values in the embedding
  NONDET    same random_state gave different output
  HANG      exceeded the timeout
  CRASH     the interpreter died (segfault / abort)
"""

import json
import subprocess
import sys
import textwrap

TIMEOUT = 120

CASES = {
    # --- duplicates / degenerate geometry ---
    "all points identical": "X = np.ones((200, 5))",
    "all identical, 1 outlier": "X = np.ones((200, 5)); X[0] += 1",
    "half duplicates": "X = np.repeat(rng.rand(100, 5), 2, axis=0)",
    "each point x30 duplicated": "X = np.repeat(rng.rand(20, 5), 30, axis=0)",
    "constant feature column": "X = np.c_[rng.rand(300, 5), np.full(300, 7.0)]",
    "single feature": "X = rng.rand(300, 1)",
    "collinear (rank 1, 50-dim)": "X = np.outer(rng.rand(300), rng.rand(50))",
    "rank-1, >100 dims (PCA path)": "X = np.outer(rng.rand(300), rng.rand(200))",
    "integer grid (many ties)": "X = rng.randint(0, 3, size=(500, 4)).astype(float)",
    # --- sizes ---
    "n=3": "X = rng.rand(3, 4)",
    "n=4": "X = rng.rand(4, 4)",
    "n=11 (n_neighbors+1)": "X = rng.rand(11, 4)",
    "n=60 (< n_neighbors+50)": "X = rng.rand(60, 4)",
    "n_components > n_features": "X = rng.rand(200, 2); kw = dict(n_components=5)",
    "n_components=1": "X = rng.rand(200, 5); kw = dict(n_components=1)",
    "n_components=10": "X = rng.rand(200, 20); kw = dict(n_components=10)",
    "n_neighbors > n_samples": "X = rng.rand(30, 4); kw = dict(n_neighbors=100)",
    "mn_ratio=0": "X = rng.rand(200, 4); kw = dict(mn_ratio=0.0)",
    "very wide (n=50, d=5000)": "X = rng.rand(50, 5000)",
    # --- scale / dtype ---
    "huge values 1e30": "X = rng.rand(300, 5) * 1e30",
    "tiny values 1e-30": "X = rng.rand(300, 5) * 1e-30",
    "mixed scale features": "X = rng.rand(300, 5) * np.array([1e-8, 1, 1e8, 1, 1])",
    "float32 input": "X = rng.rand(300, 5).astype(np.float32)",
    "int input": "X = rng.randint(0, 100, size=(300, 5))",
    "bool input": "X = rng.rand(300, 8) > 0.5",
    "Fortran-ordered": "X = np.asfortranarray(rng.rand(300, 5))",
    "read-only input": "X = rng.rand(300, 5); X.setflags(write=False)",
    "memmap input": (
        "import tempfile, os; f = tempfile.NamedTemporaryFile(delete=False); "
        "a = np.memmap(f.name, dtype=np.float64, mode='w+', shape=(300, 5)); "
        "a[:] = rng.rand(300, 5); a.flush(); "
        "X = np.memmap(f.name, dtype=np.float64, mode='r', shape=(300, 5))"
    ),
    "pandas DataFrame": "import pandas as pd; X = pd.DataFrame(rng.rand(300, 5), columns=list('abcde'))",
    "sparse CSR input": "import scipy.sparse as sp; X = sp.random(300, 50, density=0.1, format='csr', random_state=0)",
    "NaN in input": "X = rng.rand(300, 5); X[3, 2] = np.nan",
    "inf in input": "X = rng.rand(300, 5); X[3, 2] = np.inf",
    # --- metrics ---
    "angular with zero vectors": "X = rng.rand(300, 5); X[:10] = 0; kw = dict(metric='angular')",
    "angular, all same direction": "X = np.outer(rng.rand(300) + 0.1, rng.rand(5)); kw = dict(metric='angular')",
    "hamming on continuous data": "X = rng.rand(300, 5); kw = dict(metric='hamming')",
    "hamming, all-equal rows": "X = np.zeros((300, 8)); kw = dict(metric='hamming')",
    "manhattan, duplicates": "X = np.repeat(rng.rand(50, 5), 6, axis=0); kw = dict(metric='manhattan')",
    # --- precomputed ---
    "precomputed all zeros": "X = np.zeros((100, 100)); kw = dict(metric='precomputed', init='random')",
    "precomputed asymmetric": "D = rng.rand(100, 100); np.fill_diagonal(D, 0); X = D; kw = dict(metric='precomputed', init='random')",
    "precomputed nonzero diagonal": "from scipy.spatial.distance import cdist; P = rng.rand(100, 3); X = cdist(P, P) + 1; kw = dict(metric='precomputed', init='random')",
    "precomputed sparse": "import scipy.sparse as sp; from sklearn.neighbors import kneighbors_graph; X = kneighbors_graph(rng.rand(100, 3), 20, mode='distance'); kw = dict(metric='precomputed', init='random')",
    # --- optimizer / params ---
    "num_iters=0": "X = rng.rand(200, 5); kw = dict(num_iters=0)",
    "num_iters=(0,0,0)": "X = rng.rand(200, 5); kw = dict(num_iters=(0, 0, 0))",
    "num_iters=(0,0,10)": "X = rng.rand(200, 5); kw = dict(num_iters=(0, 0, 10))",
    "learning_rate=1e6": "X = rng.rand(300, 5); kw = dict(learning_rate=1e6)",
    "learning_rate=1e-12": "X = rng.rand(300, 5); kw = dict(learning_rate=1e-12)",
    "init array with constant column": "X = rng.rand(200, 5); kw = dict(init=np.c_[rng.rand(200), np.ones(200)])",
    "init array with NaN": "X = rng.rand(200, 5); I = rng.rand(200, 2); I[0, 0] = np.nan; kw = dict(init=I)",
    "init list (not ndarray)": "X = rng.rand(200, 5); kw = dict(init=rng.rand(200, 2).tolist())",
    "random_state=np.int64": "X = rng.rand(200, 5); kw = dict(random_state=np.int64(3))",
    "random_state=RandomState": "X = rng.rand(200, 5); kw = dict(random_state=np.random.RandomState(3))",
    # --- larger ---
    "n=100k, d=10": "X = rng.rand(100_000, 10).astype(np.float32)",
}

RUNNER = textwrap.dedent(
    """
    import json, sys, warnings
    import numpy as np
    from pacmap_sklearn import PaCMAP, LocalMAP
    rng = np.random.RandomState(0)
    kw = {}
    %s
    Est = {"PaCMAP": PaCMAP, "LocalMAP": LocalMAP}[sys.argv[1]]
    base = dict(random_state=0)
    base.update(kw)
    if "num_iters" not in kw and getattr(X, "shape", (0,))[0] >= 50000:
        base["num_iters"] = (10, 10, 10)
    if isinstance(base.get("random_state"), np.random.RandomState):
        mk = lambda: Est(**{**base, "random_state": np.random.RandomState(3)})
    else:
        mk = lambda: Est(**base)
    out = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            Y1 = mk().fit_transform(X)
        except Exception as e:
            out = {"status": "ERROR", "msg": f"{type(e).__name__}: {e}"[:200]}
        else:
            Y2 = mk().fit_transform(X)
            n = X.shape[0]
            if Y1.shape != (n, base.get("n_components", 2)):
                out = {"status": "SHAPE", "msg": str(Y1.shape)}
            elif not np.all(np.isfinite(Y1)):
                out = {"status": "NAN", "msg": f"{(~np.isfinite(Y1)).sum()} non-finite"}
            elif not np.array_equal(Y1, Y2):
                out = {"status": "NONDET", "msg": f"max diff {np.abs(Y1 - Y2).max():.3g}"}
            else:
                spread = float(np.ptp(Y1, axis=0).min()) if n > 1 else 0.0
                out = {"status": "ok", "msg": f"min axis spread {spread:.3g}"}
        out["warnings"] = sorted({f"{x.category.__name__}: {str(x.message)[:90]}" for x in w})
    print("RESULT" + json.dumps(out))
    """
)


def run(case, code, est):
    script = RUNNER % code
    try:
        p = subprocess.run(
            [sys.executable, "-c", script, est],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"status": "HANG", "msg": f">{TIMEOUT}s"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[6:])
    return {"status": "CRASH", "msg": f"exit {p.returncode}: {p.stderr.strip()[-200:]}"}


if __name__ == "__main__":
    only = sys.argv[1:]
    for est in ("PaCMAP", "LocalMAP"):
        print(f"\n=== {est} ===")
        for case, code in CASES.items():
            if only and not any(o in case for o in only):
                continue
            r = run(case, code, est)
            warn = f"  [warn: {'; '.join(r.get('warnings', []))}]" if r.get("warnings") else ""
            print(f"{r['status']:7s} {case:34s} {r['msg']}{warn}", flush=True)
