"""Tests for the transform feature of pacmap, with the trees"""

import pytest
from test_transform import mnist_data, setup_transform_test


def test_transform_tree_2_splits(mnist_data):
    """Test transform feature with trees and 2 splits"""
    setup_transform_test(mnist_data, 2, save_tree=True)


def test_transform_tree_5_splits(mnist_data):
    """Test transform feature with trees and 5 splits"""
    setup_transform_test(mnist_data, 5, save_tree=True)


def test_transform_tree_10_splits(mnist_data):
    """Test transform feature with trees and 10 splits"""
    setup_transform_test(mnist_data, 10, save_tree=True)

