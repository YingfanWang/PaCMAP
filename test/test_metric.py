'''
A test script that ensures PaCMAP can be successfully used with other metrics.
'''
import pacmap
import numpy as np
from test_utils import generate_figure
from test_general import debug_nondeterminism
import pytest


def test_pacmap_metric_initialization():
    """Test that PaCMAP can be initialized with different distance metrics."""
    # Try initialize
    pacmap.PaCMAP(distance='manhattan')
    pacmap.PaCMAP(distance='angular')
    pacmap.PaCMAP(distance='hamming')

def test_pacmap_metric_unknown_fails():
    """Test that PaCMAP fails to initialize with unknown distance metrics."""
    with pytest.raises(NotImplementedError):
        pacmap.PaCMAP(distance='unknown')


def test_pacmap_metric_deterministic_manhattan():
    """Test PaCMAP deterministic behavior with different metrics."""
    # Initialize sample data
    sample_data = np.random.normal(size=(10000, 10))
    b = pacmap.PaCMAP(random_state=10, distance='manhattan')
    b_out = b.fit_transform(sample_data)
    c = pacmap.PaCMAP(random_state=10, distance='manhattan')
    c_out = c.fit_transform(sample_data)
    try:
        assert np.allclose(b_out, c_out, atol=1e-8)
    except AssertionError:
        # Print debug output and re-raise error.
        debug_nondeterminism(b, c)
        raise AssertionError

def test_pacmap_metric_angular():
    """Test PaCMAP works with angular distance metric."""
    sample_data = np.random.normal(size=(10000, 10))
    d = pacmap.PaCMAP(distance='angular')
    d_out = d.fit_transform(sample_data)

def test_pacmap_metric_hamming():
    """Test PaCMAP works with hamming distance metric."""
    sample_data = np.random.normal(size=(10000, 10))
    e = pacmap.PaCMAP(distance='hamming', apply_pca=False)
    e_out = e.fit_transform(sample_data, init='random')

def test_pacmap_fashion_mnist_manhattan(openml_datasets):
    """Test PaCMAP with Fashion-MNIST using Manhattan distance."""
    # Load Fashion-MNIST from fixture
    fmnist, labels = openml_datasets["Fashion-MNIST"]
    fmnist = fmnist.reshape(fmnist.shape[0], -1)
    
    # Use subset for faster testing
    fmnist = fmnist[:1000]
    labels = labels[:1000].astype(int)
    
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, distance='manhattan')
    embedding = reducer.fit_transform(fmnist)
    generate_figure(embedding, labels, 'test_fmnist_manhattan')

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, distance='manhattan')
    embedding = reducer.fit_transform(fmnist)
    generate_figure(embedding, labels, 'test_fmnist_manhattan_noseed')


def test_pacmap_mnist_metrics(openml_datasets):
    """Test PaCMAP with MNIST using different distance metrics."""
    # Load MNIST from fixture
    mnist, labels = openml_datasets["mnist_784"]
    mnist = mnist.reshape(mnist.shape[0], -1)
    
    # Use subset for faster testing
    mnist = mnist[:1000]
    labels = labels[:1000].astype(int)
    
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, distance='angular')
    embedding = reducer.fit_transform(mnist)
    generate_figure(embedding, labels, 'test_mnist_angular')

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, distance='hamming', apply_pca=True)
    embedding = reducer.fit_transform(mnist)
    generate_figure(embedding, labels, 'test_mnist_hamming')

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, distance='manhattan')
    embedding = reducer.fit_transform(mnist)
    generate_figure(embedding, labels, 'test_mnist_manhattan')

    print('Figures have been generated successfully.')


if __name__ == "__main__":
    # Backward compatibility - can still run as script
    test_pacmap_metric_initialization()
    test_pacmap_metric_deterministic()
    test_pacmap_fashion_mnist_manhattan()
    test_pacmap_mnist_metrics()
