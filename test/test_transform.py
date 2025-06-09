"""Tests for the transform feature of pacmap"""

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


def setup_transform_test(mnist_data, n_splits, save_tree=False):
    """Setup test data and run PaCMAP transform"""
    print("Test start")
    mnist, labels = mnist_data
    skf = StratifiedKFold(n_splits=n_splits)
    train_index, test_index = next(skf.split(mnist, labels))
    X_train, X_test = mnist[train_index], mnist[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, save_tree=save_tree, verbose=True)
    embedding = reducer.fit_transform(X_train)
    if save_tree:
        embedding_extra = reducer.transform(X_test)
    else:
        embedding_extra = reducer.transform(X_test, basis=X_train)
    embedding_combined = np.concatenate((embedding, embedding_extra))
    y = np.concatenate((y_train, y_test))
    embeddings = [embedding, embedding_extra, embedding_combined]
    labelset = [y_train, y_test, y]
    titles = [f'basis_{n_splits}', f'extend_{n_splits}', f'full_{n_splits}']
    suffix = '_tree' if save_tree else ''
    generate_combined_figure(embeddings, labelset, titles, f'test_mnist_transform_{n_splits}{suffix}')

    # Basic assertions to verify the transform worked correctly
    assert embedding.shape[0] == X_train.shape[0], "Training embedding should have same number of samples as training data"
    assert embedding.shape[1] == 2, "Embedding should be 2-dimensional"
    assert embedding_extra.shape[0] == X_test.shape[0], "Test embedding should have same number of samples as test data"
    assert embedding_extra.shape[1] == 2, "Test embedding should be 2-dimensional"
    assert embedding_combined.shape[0] == X_train.shape[0] + X_test.shape[0], "Combined embedding should have all samples"
    assert not np.any(np.isnan(embedding)), "Training embedding should not contain NaN values"
    assert not np.any(np.isnan(embedding_extra)), "Test embedding should not contain NaN values"

    # Print distance information for debugging/verification
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


def test_transform_2_splits(mnist_data):
    """Test transform feature with 2 splits"""
    setup_transform_test(mnist_data, 2)


def test_transform_5_splits(mnist_data):
    """Test transform feature with 5 splits"""
    setup_transform_test(mnist_data, 5)


def test_transform_10_splits(mnist_data):
    """Test transform feature with 10 splits"""
    setup_transform_test(mnist_data, 10)
