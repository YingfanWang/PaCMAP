'''A script that tests the transform feature of pacmap
'''

import pacmap
import numpy as np
from sklearn.model_selection import StratifiedKFold
from test_utils import *


if __name__ == "__main__":
    # MNIST
    print("Test start")
    # Please manually extract from the data folder
    mnist = np.load("../data/mnist_images.npy", allow_pickle=True)
    mnist = mnist.reshape(mnist.shape[0], -1)
    labels = np.load("../data/mnist_labels.npy", allow_pickle=True)
    print("Loading data")
    n_splits = [2, 5, 10]
    for n in n_splits:
        skf = StratifiedKFold(n_splits=n)
        for train_index, test_index in skf.split(mnist, labels):
            X_train, X_test = mnist[train_index], mnist[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            break
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, save_tree=False, verbose=True)
        embedding = reducer.fit_transform(X_train)
        embedding_extra = reducer.transform(X_test, basis=X_train)
        embedding_combined = np.concatenate((embedding, embedding_extra))
        y = np.concatenate((y_train, y_test))
        embeddings = [embedding, embedding_extra, embedding_combined]
        labelset = [y_train, y_test, y]
        titles = [f'basis_{n}', f'extend_{n}', f'full_{n}']
        generate_combined_figure(embeddings, labelset, titles, f'test_mnist_transform_{n}')
        for i in range(10):
            xp0 = reducer.pair_XP[i*100][0]
            xp1 = reducer.pair_XP[i*100][1]
            dist = np.linalg.norm(X_test[xp0-len(X_train)] - X_train[xp1])
            print(y[xp0], y[xp1], dist)
        for i in range(10):
            xp0 = reducer.pair_neighbors[i*100][0]
            xp1 = reducer.pair_neighbors[i*100][1]
            dist = np.linalg.norm(X_train[xp0] - X_train[xp1])
            print(y[xp0], y[xp1], dist)
