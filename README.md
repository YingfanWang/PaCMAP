# PaCMAP

PaCMAP (Pairwise Controlled Manifold Approximation) is a dimensionality reduction method that can be used for visualization, preserving both local and global structure of the data in original space. PaCMAP optimizes the low dimensional embedding using three kinds of pairs of points: neighbor pairs (pair_neighbors), mid-near pair (pair_MN), and further pairs (pair_FP), whose numbers are n_neighbors, n_MN and n_FP respectively.

Previous dimensionality reduction techniques focus on either local structure (e.g. t-SNE, LargeVis and UMAP) or global structure (e.g. TriMAP), but not both, although with carefully parameter tuning. 

# Installation

To install PaCMAP, you can use pip:

`pip install pacmap`

# Benchmarks

List some benchmarks and examples for PaCMAP

# Parameters

The list of (important) parameters is given below.

- n_neighbors
n_neighbors controls the number of neighbors that are considered in the k-Nearest Neighbor graph

- mid_ratio

- FP_ratio



# Citation

# License

Please see the license file.