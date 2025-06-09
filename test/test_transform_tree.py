"""Tests for the transform feature of pacmap, with the trees"""

import pacmap
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_openml
from test_utils import generate_combined_figure


@pytest.fixture(scope="module")
def mnist_data():
    """Load MNIST data for testing"""
    print("Loading data")
    # Load MNIST from OpenML
    mnist_data = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    mnist, labels = mnist_data
    mnist = mnist.reshape(mnist.shape[0], -1)
    
    # Use subset for faster testing
    mnist = mnist[:5000]
    labels = labels[:5000].astype(int)
    return mnist, labels


def test_transform_tree_2_splits(mnist_data):
    """Test transform feature with trees and 2 splits"""
    print("Test start")
    mnist, labels = mnist_data
    n = 2
    skf = StratifiedKFold(n_splits=n)
    train_index, test_index = next(skf.split(mnist, labels))
    X_train, X_test = mnist[train_index], mnist[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, save_tree=True, verbose=True)
    embedding = reducer.fit_transform(X_train)
    embedding_extra = reducer.transform(X_test)
    embedding_combined = np.concatenate((embedding, embedding_extra))
    y = np.concatenate((y_train, y_test))
    embeddings = [embedding, embedding_extra, embedding_combined]
    labelset = [y_train, y_test, y]
    titles = [f'basis_{n}', f'extend_{n}', f'full_{n}']
    generate_combined_figure(embeddings, labelset, titles, f'test_mnist_transform_{n}_tree')
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


def test_transform_tree_5_splits(mnist_data):
    """Test transform feature with trees and 5 splits"""
    print("Test start")
    mnist, labels = mnist_data
    n = 5
    skf = StratifiedKFold(n_splits=n)
    train_index, test_index = next(skf.split(mnist, labels))
    X_train, X_test = mnist[train_index], mnist[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, save_tree=True, verbose=True)
    embedding = reducer.fit_transform(X_train)
    embedding_extra = reducer.transform(X_test)
    embedding_combined = np.concatenate((embedding, embedding_extra))
    y = np.concatenate((y_train, y_test))
    embeddings = [embedding, embedding_extra, embedding_combined]
    labelset = [y_train, y_test, y]
    titles = [f'basis_{n}', f'extend_{n}', f'full_{n}']
    generate_combined_figure(embeddings, labelset, titles, f'test_mnist_transform_{n}_tree')
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


def test_transform_tree_10_splits(mnist_data):
    """Test transform feature with trees and 10 splits"""
    print("Test start")
    mnist, labels = mnist_data
    n = 10
    skf = StratifiedKFold(n_splits=n)
    train_index, test_index = next(skf.split(mnist, labels))
    X_train, X_test = mnist[train_index], mnist[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, save_tree=True, verbose=True)
    embedding = reducer.fit_transform(X_train)
    embedding_extra = reducer.transform(X_test)
    embedding_combined = np.concatenate((embedding, embedding_extra))
    y = np.concatenate((y_train, y_test))
    embeddings = [embedding, embedding_extra, embedding_combined]
    labelset = [y_train, y_test, y]
    titles = [f'basis_{n}', f'extend_{n}', f'full_{n}']
    generate_combined_figure(embeddings, labelset, titles, f'test_mnist_transform_{n}_tree')
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

