'''
A script that tests the transform feature of pacmap
'''

import pacmap
import numpy as np
import pytest
from test_utils import generate_combined_figure
from sklearn import datasets


@pytest.fixture
def iris_data():
    """Load iris dataset for testing."""
    iris = datasets.load_iris()
    return iris["data"], iris["target"]


def test_iris_transform_with_tree(iris_data):
    """Test PaCMAP transform functionality with save_tree=True."""
    iris, label = iris_data

    reducer = pacmap.PaCMAP(save_tree=True, verbose=True)
    embedding = reducer.fit_transform(iris)
    embedding_extra = reducer.transform(iris)

    # Verify embeddings have correct shape
    assert embedding.shape == (len(iris), 2)
    assert embedding_extra.shape == (len(iris), 2)

    # Generate visualization
    embedding_combined = np.concatenate((embedding, embedding_extra))
    embeddings = [embedding, embedding_extra, embedding_combined]
    y = np.concatenate((label, label))
    labelset = [label, label, y]
    titles = [f'basis', f'extend', f'full']
    generate_combined_figure(embeddings, labelset, titles, f'test_iris_transform_tree')

    # Test pair relationships
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


def test_iris_transform_without_tree(iris_data):
    """Test PaCMAP transform functionality with save_tree=False."""
    iris, label = iris_data

    reducer = pacmap.PaCMAP(save_tree=False, verbose=True)
    embedding = reducer.fit_transform(iris)
    embedding_extra = reducer.transform(iris, basis=iris)

    # Verify embeddings have correct shape
    assert embedding.shape == (len(iris), 2)
    assert embedding_extra.shape == (len(iris), 2)

    # Generate visualization
    embedding_combined = np.concatenate((embedding, embedding_extra))
    embeddings = [embedding, embedding_extra, embedding_combined]
    y = np.concatenate((label, label))
    labelset = [label, label, y]
    titles = [f'basis', f'extend', f'full']
    generate_combined_figure(embeddings, labelset, titles, f'test_iris_transform')

    # Test pair relationships
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


if __name__ == "__main__":
    # Backward compatibility - can still run as script
    pytest.main([__file__])
