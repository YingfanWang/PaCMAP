import pacmap
import numpy as np
import matplotlib.pyplot as plt

# loading preprocessed coil_20 dataset
# you can change it with any dataset that is in the ndarray format, with the shape (N, D)
# where N is the number of samples and D is the dimension of each sample
X = np.load("../data/coil_20.npy", allow_pickle=True)
X = X.reshape(X.shape[0], -1)
y = np.load("./data/coil_20_labels.npy", allow_pickle=True)

# Initialize the pacmap instance
# By default, the n_neighbors is set to 10.
# Setting n_neighbors to "None" can enable an automatic parameter selection
# choice shown in "parameter" section of the README file. 
# Notice that from v0.6.0 on, we rename the n_dims parameter to n_components.
reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0) 

# fit the data (The index of transformed data corresponds to the index of the original data)
X_transformed = reducer.fit_transform(X, init="pca")

# visualize the embedding
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)

# saving the reducer
pacmap.save(reducer, "./coil_20_reducer")

# loading the reducer
pacmap.load("./coil_20_reducer")
