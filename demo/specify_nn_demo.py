import pacmap
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex

# loading preprocessed coil_20 dataset
X = np.load("../data/coil_20.npy", allow_pickle=True)
X = X.reshape(X.shape[0], -1)
y = np.load("../data/coil_20_labels.npy", allow_pickle=True)

# create nearest neighbor pairs
# here we use AnnoyIndex as an example, but the process can be done by any
# external NN library that provides neighbors into a matrix of the shape
# (n, n_neighbors_extra), where n_neighbors_extra is greater or equal to
# n_neighbors in the following example.

n, dim = X.shape
n_neighbors = 10
tree = AnnoyIndex(dim, metric='euclidean')
for i in range(n):
    tree.add_item(i, X[i, :])
tree.build(20)

nbrs = np.zeros((n, 20), dtype=np.int32)
for i in range(n):
    nbrs_ = tree.get_nns_by_item(i, 20 + 1) # The first nbr is always the point itself
    nbrs[i, :] = nbrs_[1:]

scaled_dist = np.ones((n, n_neighbors)) # No scaling is needed

# Type casting is needed for numba acceleration
X = X.astype(np.float32)
scaled_dist = scaled_dist.astype(np.float32)

# make sure n_neighbors is the same number you want when fitting the data
pair_neighbors = pacmap.sample_neighbors_pair(X, scaled_dist, nbrs, np.int32(n_neighbors))

# initializing the pacmap instance
# feed the pair_neighbors into the instance
embedding = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, MN_ratio=0.5, FP_ratio=2.0, pair_neighbors=pair_neighbors) 

# fit the data (The index of transformed data corresponds to the index of the original data)
X_transformed = embedding.fit_transform(X, init="pca")

# visualize the embedding
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)
