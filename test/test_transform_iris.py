'''
A script that tests the transform feature of pacmap
'''

import pacmap
import numpy as np
from sklearn.model_selection import StratifiedKFold
from test_utils import *
from sklearn import datasets


if __name__ == "__main__":
    # Iris
    iris = datasets.load_iris()['data']
    label = datasets.load_iris()['target']
    reducer = pacmap.PaCMAP(save_tree=True, verbose=True)
    embedding = reducer.fit_transform(iris)
    embedding_extra = reducer.transform(iris)
    embedding_combined = np.concatenate((embedding, embedding_extra))
    embeddings = [embedding, embedding_extra, embedding_combined]
    y = np.concatenate((label, label))
    labelset = [label, label, y]
    titles = [f'basis', f'extend', f'full']
    generate_combined_figure(embeddings, labelset, titles, f'test_iris_transform_tree')
    for i in range(10):
        xp0 = reducer.pair_XP[i*100][0]
        xp1 = reducer.pair_XP[i*100][1]
        dist = np.linalg.norm(iris[xp0-len(iris)] - iris[xp1])
        print(y[xp0], y[xp1], dist)
    for i in range(10):
        xp0 = reducer.pair_neighbors[i*100][0]
        xp1 = reducer.pair_neighbors[i*100][1]
        dist = np.linalg.norm(iris[xp0] - iris[xp1])
        print(y[xp0], y[xp1], dist)

    reducer = pacmap.PaCMAP(save_tree=False, verbose=True)
    embedding = reducer.fit_transform(iris)
    embedding_extra = reducer.transform(iris, basis=iris)
    embedding_combined = np.concatenate((embedding, embedding_extra))
    embeddings = [embedding, embedding_extra, embedding_combined]
    labelset = [label, label, y]
    titles = [f'basis', f'extend', f'full']
    generate_combined_figure(embeddings, labelset, titles, f'test_iris_transform')
    for i in range(10):
        xp0 = reducer.pair_XP[i*100][0]
        xp1 = reducer.pair_XP[i*100][1]
        dist = np.linalg.norm(iris[xp0-len(iris)] - iris[xp1])
        print(y[xp0], y[xp1], dist)
    for i in range(10):
        xp0 = reducer.pair_neighbors[i*100][0]
        xp1 = reducer.pair_neighbors[i*100][1]
        dist = np.linalg.norm(iris[xp0] - iris[xp1])
        print(y[xp0], y[xp1], dist)
