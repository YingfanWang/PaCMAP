# PaCMAP

PaCMAP (Pairwise Controlled Manifold Approximation) is a dimensionality reduction method that can be used for visualization, preserving both local and global structure of the data in original space. PaCMAP optimizes the low dimensional embedding using three kinds of pairs of points: neighbor pairs (pair_neighbors), mid-near pair (pair_MN), and further pairs (pair_FP), whose numbers are n_neighbors, n_MN and n_FP respectively.

Previous dimensionality reduction techniques focus on either local structure (e.g. t-SNE, LargeVis and UMAP) or global structure (e.g. TriMAP), but not both, although with carefully tuning the parameter in their algorithms that controls the balance between global and local structure, which mainly adjusts the number of considered neighbors. Instead of considering more neighbors to attract for preserving glocal structure, PaCMAP dynamically uses a special group of pairs -- mid-near pairs, to first capture global structure and then refine local structure, which both preserve global and local structure.

# Installation
Requirements:
- numpy
- sklearn
- annoy
- numba

To install PaCMAP, you can use pip:

`pip install pacmap`

# Benchmarks

The following are DR algorithms' performance on two datasets: MNIST and Mammoth that repectively demonstrate local and global structure's preservation:
![image](http://github.com/YingfanWang/PaCMAP/raw/master/images/MNIST.jpg)


# Parameters

The list of (important) parameters is given below.

- n_neighbors: n_neighbors controls the number of neighbors considered in the k-Nearest Neighbor graph

- MN_ratio: the ratio of the number of mid-near pairs to the number of neighbors, n_MN = n_neighbors * MN_ratio // 1

- FP_ratio: the ratio of the number of further pairs to the number of neighbors, n_FP = n_neighbors * FP_ratio // 1



# Citation

# License

Please see the license file.
