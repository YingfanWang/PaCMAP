"""Figures for bench_vs_tsne.py output.

    python sklearn_port/benchmarks/plot_vs_tsne.py RESULTS.json OUT_PREFIX
"""

import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLORS = {  # categorical slots 1-3, fixed order
    "PaCMAP": "#2a78d6",
    "t-SNE (PCA init)": "#eb6834",
    "t-SNE (random init)": "#1baf7a",
}
MARKERS = {"PaCMAP": "o", "t-SNE (PCA init)": "s", "t-SNE (random init)": "^"}
METRICS = [
    ("trust", "Trustworthiness@10\n(local, ↑)", False),
    ("knn", "10-NN accuracy\n(local, ↑)", False),
    ("triplet", "Random-triplet accuracy\n(global, ↑)", False),
    ("dist_corr", "Distance Spearman ρ\n(global, ↑)", False),
    ("centroid_corr", "Class-centroid Spearman ρ\n(global, ↑)", False),
    ("time", "Fit time [s]\n(log scale, ↓)", True),
]


def aggregate(results):
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        for k, *_ in METRICS:
            if k in r and r[k] is not None:
                agg[(r["dataset"], r["method"])][k].append(r[k])
    datasets = list(dict.fromkeys(r["dataset"] for r in results))
    return agg, datasets


def metrics_figure(results, out):
    agg, datasets = aggregate(results)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(22, 0.55 * len(datasets) + 2.2),
                             sharey=True, facecolor=SURFACE)
    ypos = np.arange(len(datasets))[::-1]
    for ax, (key, title, log) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for j, (method, color) in enumerate(COLORS.items()):
            xs, ys, errs = [], [], []
            for d, y in zip(datasets, ypos):
                v = agg[(d, method)].get(key)
                if v:
                    xs.append(np.mean(v)); ys.append(y + (1 - j) * 0.18)
                    errs.append(np.ptp(v) / 2)
            ax.errorbar(xs, ys, xerr=errs, fmt=MARKERS[method], color=color, ms=8,
                        mec=SURFACE, mew=1.5, elinewidth=2, capsize=0, label=method)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        if log:
            ax.set_xscale("log")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for y in ypos[1::2]:
            ax.axhspan(y - 0.5, y + 0.5, color="#f3f2ee", zorder=0)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(datasets, color=INK, fontsize=9.5)
    axes[0].set_ylim(-0.6, len(datasets) - 0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False,
               bbox_to_anchor=(0.005, 0.995), fontsize=10, labelcolor=INK)
    fig.text(0.995, 0.985, "Mean over seeds; bars show min–max. Missing points: dataset has no labels.",
             ha="right", va="top", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110, facecolor=SURFACE)


def embeddings_figure(npz_path, datasets, out):
    E = np.load(npz_path, allow_pickle=False)
    fig, axes = plt.subplots(len(datasets), 3, figsize=(10.5, 3.3 * len(datasets)),
                             facecolor=SURFACE)
    for i, d in enumerate(datasets):
        labels = E[f"{d}|labels"] if f"{d}|labels" in E.files else None
        for j, method in enumerate(COLORS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            key = f"{d}|{method}"
            if key in E.files:
                Y = E[key]
                if labels is not None:
                    c, cmap = labels, "tab20" if len(np.unique(labels)) > 10 else "tab10"
                else:  # unlabeled: color by the first input-space coordinate order
                    c, cmap = np.arange(len(Y)), "viridis"
                ax.scatter(Y[:, 0], Y[:, 1], c=c, cmap=cmap, s=0.6 if len(Y) > 5000 else 2,
                           linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if i == 0:
                ax.set_title(method, fontsize=11, color=INK)
            if j == 0:
                ax.set_ylabel(d, fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=90, facecolor=SURFACE)


if __name__ == "__main__":
    results = json.load(open(sys.argv[1]))
    prefix = sys.argv[2]
    _, datasets = aggregate(results)
    metrics_figure(results, prefix + "_metrics.png")
    embeddings_figure(sys.argv[1].replace(".json", ".npz"), datasets,
                      prefix + "_embeddings.png")
