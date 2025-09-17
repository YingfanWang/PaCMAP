'''A script that tests the transform feature of pacmap, with the trees
'''

import numpy as np
from test_transform import mnist_data, setup_transform_test


def test_transform_tree_2_splits(mnist_data):
    """Test transform feature with trees and 2 splits"""
    X_train, X_test, embeddings = setup_transform_test(mnist_data, 2, save_tree=True)
    [embedding, embedding_extra, embedding_combined] = embeddings

    # Basic assertions to verify the transform worked correctly
    assert embedding.shape[0] == X_train.shape[0], "Training embedding should have same number of samples as training data"
    assert embedding.shape[1] == 2, "Embedding should be 2-dimensional"
    assert embedding_extra.shape[0] == X_test.shape[0], "Test embedding should have same number of samples as test data"
    assert embedding_extra.shape[1] == 2, "Test embedding should be 2-dimensional"
    assert embedding_combined.shape[0] == X_train.shape[0] + X_test.shape[0], "Combined embedding should have all samples"
    assert not np.any(np.isnan(embedding)), "Training embedding should not contain NaN values"
    assert not np.any(np.isnan(embedding_extra)), "Test embedding should not contain NaN values"


def test_transform_tree_5_splits(mnist_data):
    """Test transform feature with trees and 5 splits"""
    X_train, X_test, embeddings = setup_transform_test(mnist_data, 5, save_tree=True)
    [embedding, embedding_extra, embedding_combined] = embeddings

    # Basic assertions to verify the transform worked correctly
    assert embedding.shape[0] == X_train.shape[0], "Training embedding should have same number of samples as training data"
    assert embedding.shape[1] == 2, "Embedding should be 2-dimensional"
    assert embedding_extra.shape[0] == X_test.shape[0], "Test embedding should have same number of samples as test data"
    assert embedding_extra.shape[1] == 2, "Test embedding should be 2-dimensional"
    assert embedding_combined.shape[0] == X_train.shape[0] + X_test.shape[0], "Combined embedding should have all samples"
    assert not np.any(np.isnan(embedding)), "Training embedding should not contain NaN values"
    assert not np.any(np.isnan(embedding_extra)), "Test embedding should not contain NaN values"


def test_transform_tree_10_splits(mnist_data):
    """Test transform feature with trees and 10 splits"""
    X_train, X_test, embeddings = setup_transform_test(mnist_data, 10, save_tree=True)
    [embedding, embedding_extra, embedding_combined] = embeddings

    # Basic assertions to verify the transform worked correctly
    assert embedding.shape[0] == X_train.shape[0], "Training embedding should have same number of samples as training data"
    assert embedding.shape[1] == 2, "Embedding should be 2-dimensional"
    assert embedding_extra.shape[0] == X_test.shape[0], "Test embedding should have same number of samples as test data"
    assert embedding_extra.shape[1] == 2, "Test embedding should be 2-dimensional"
    assert embedding_combined.shape[0] == X_train.shape[0] + X_test.shape[0], "Combined embedding should have all samples"
    assert not np.any(np.isnan(embedding)), "Training embedding should not contain NaN values"
    assert not np.any(np.isnan(embedding_extra)), "Test embedding should not contain NaN values"

