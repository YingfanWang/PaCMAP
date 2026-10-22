"""Side-by-side embeddings: port vs reference (vs sklearn TSNE).

    PYTHONPATH=sklearn_port python sklearn_port/benchmarks/plot_embeddings.py OUT.png

Same datasets and seed as bench_pacmap.py. Classes beyond the 8 palette hues
reuse a hue with a different marker shape, and each class is labelled at
its median position so colour is never the only identifier.
"""

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bench_pacmap import load, scores, synthetic
from sklearn.manifold import TSNE

import pacmap.pacmap as ref
from pacmap_sklearn import PaCMAP

HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#6250d6", "#e34948"]
SHAPES = ["o", "^", "s"]  # slot k uses hue k % 8, shape k // 8
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def draw(ax, emb, y, title, metrics, seconds):
    classes = np.unique(y)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(emb))  # no class is always painted on top
    hue = {c: HUES[k % 8] for k, c in enumerate(classes)}
    shape = {c: SHAPES[k // 8] for k, c in enumerate(classes)}
    size = 3 if len(emb) < 20000 else 0.6
    for s in SHAPES:
        idx = order[np.isin(y[order], [c for c in classes if shape[c] == s])]
        if len(idx):
            ax.scatter(emb[idx, 0], emb[idx, 1], s=size, marker=s, linewidths=0,
                       c=[hue[c] for c in y[idx]], alpha=0.75, rasterized=True)
    for c in classes:
        mx, my = np.median(emb[y == c], axis=0)
        ax.text(mx, my, str(c), fontsize=7, color=INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.text(0.0, -0.02, f"trust {metrics['trust']:.3f}   10-NN {metrics['knn']:.3f}   "
            f"triplet {metrics['triplet']:.3f}   {seconds:.1f}s",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#e1e0d9")
    ax.set_facecolor(SURFACE)


def main(out):
    datasets = [("USPS", *load("USPS"), True),
                ("20NG", *load("20NG"), True),
                ("Synthetic 70k", *synthetic(), False)]
    ref.PaCMAP(random_state=0).fit_transform(datasets[0][1][:200])  # JIT warm-up
    fig, axes = plt.subplots(3, 3, figsize=(13, 13), facecolor=SURFACE)
    for row, (name, X, y, with_tsne) in enumerate(datasets):
        methods = [("port", lambda: PaCMAP(random_state=0)),
                   ("reference", lambda: ref.PaCMAP(random_state=0, knn_backend="faiss"))]
        if with_tsne:
            methods.append(("sklearn TSNE", lambda: TSNE(random_state=0)))
        for col in range(3):
            if col >= len(methods):
                axes[row, col].axis("off")
                continue
            label, make = methods[col]
            t0 = time.perf_counter()
            emb = make().fit_transform(X)
            dt = time.perf_counter() - t0
            draw(axes[row, col], emb, y, f"{name} · {label}", scores(X, y, emb, 0), dt)
            print(name, label, f"{dt:.1f}s", flush=True)
    fig.suptitle("PaCMAP: scikit-learn port vs reference (seed 0)",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.955, "Numbers mark each class's median position. trust = trustworthiness@10 "
             "(3k subsample); 10-NN = 5-fold kNN accuracy; triplet = random-triplet agreement.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1])
