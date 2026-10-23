"""PaCMAP / LocalMAP, with and without Rashomon neighbor refinement.

    PYTHONPATH=sklearn_port:sklearn_port/benchmarks \
        python sklearn_port/benchmarks/plot_variants.py OUT.png
"""

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bench_pacmap import load, scores
from plot_embeddings import INK, INK2, SURFACE, draw

from pacmap_sklearn import PaCMAP


class _LocalMAPWithRashomon(PaCMAP):
    """Reproduces the evidence for banning localmap + n_rashomon; the public
    estimator raises on this combination. Do not use for real embeddings."""

    def _check_switches(self):
        pass


def purity(est, y):
    P = est.pair_neighbors_
    return np.mean(y[P[:, 0]] == y[P[:, 1]])


def main(out, n_rashomon=5):
    variants = [
        ("PaCMAP", lambda: PaCMAP(random_state=0)),
        (f"PaCMAP · n_rashomon={n_rashomon}",
         lambda: PaCMAP(n_rashomon=n_rashomon, random_state=0)),
        ("LocalMAP", lambda: PaCMAP(localmap=True, random_state=0)),
        (f"LocalMAP · n_rashomon={n_rashomon}",
         lambda: _LocalMAPWithRashomon(localmap=True, n_rashomon=n_rashomon,
                                       random_state=0)),
    ]
    datasets = [("USPS", *load("USPS")), ("20NG", *load("20NG"))]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.6), facecolor=SURFACE)
    for row, (name, X, y) in enumerate(datasets):
        for col, (label, make) in enumerate(variants):
            t0 = time.perf_counter()
            est = make().fit(X)
            dt = time.perf_counter() - t0
            ax = axes[row, col]
            draw(ax, est.embedding_, y, f"{name} · {label}",
                 scores(X, y, est.embedding_, 0), dt)
            ax.text(0.0, -0.075, f"NN-pair label purity {purity(est, y):.3f}",
                    transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
            print(name, label, f"{dt:.1f}s", flush=True)
    fig.suptitle("Rashomon trustworthy-neighbor refinement (seed 0)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.935, "Numbers mark each class's median position. NN-pair label purity = "
             "fraction of nearest-neighbor pairs whose endpoints share a label.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3)
    fig.savefig(out, dpi=120, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1])
