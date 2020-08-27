import os
import json
import numpy as np
import matplotlib.pyplot as plt

from run_script import data_prep
from fa2 import ForceAtlas2 as FA2
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def transform_by_FA2(X, n_neighbors=6):
    if X.shape[1] > 100:
        p = PCA(n_components=100)
        X = p.fit_transform(X)
    n = NearestNeighbors(n_neighbors=n_neighbors)
    n.fit(X)
    
    # Construct the graph
    A = n.kneighbors_graph(X)
    A += A.T
    A = (A > 0).astype(int)

    # Initialize the graph.
    # Rescale to match FA2 requirements

    p = PCA(n_components=2)
    X = p.fit_transform(X)
    X_init = X[:, :2]
    X_init *= 10000/np.std(X_init)

    f = FA2()
    X_low = f.forceatlas2(A, X_init, 100)
    X_low = np.array(X_low)
    return X_low


X, y = data_prep('coil_20', 70000)
X_low = transform_by_FA2(X)
np.save('coil_20_fa2.npy', X_low)

X, y = data_prep('MNIST', 70000)
X_low = transform_by_FA2(X)
np.save('MNIST_fa2.npy', X_low)


X, y = data_prep('mammoth', 10000)
X_low = transform_by_FA2(X)
np.save('mammoth_fa2.npy', X_low)

X, y = data_prep('s_curve_hole', 10000)
X_low = transform_by_FA2(X)
np.save('s_curve_hole_fa2.npy', X_low)
