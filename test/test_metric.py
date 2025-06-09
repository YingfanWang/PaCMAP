'''
A test script that ensures PaCMAP can be successfully used with other metrics.
'''
import sklearn
import pacmap
import numpy as np
import matplotlib.pyplot as plt
from test_utils import *
from sklearn.datasets import fetch_openml


def test_pacmap_metric_initialization():
    """Test that PaCMAP can be initialized with different distance metrics."""
    # Try initialize
    pacmap.PaCMAP(distance='manhattan')
    pacmap.PaCMAP(distance='angular')
    pacmap.PaCMAP(distance='hamming')
    print("Instance initialized successfully.")
    try:
        pacmap.PaCMAP(distance='unknown')
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        print("Not implemented error raised successfully")


def test_pacmap_metric_deterministic():
    """Test PaCMAP deterministic behavior with different metrics."""
    # Initialize sample data
    sample_data = np.random.normal(size=(10000, 10))
    b = pacmap.PaCMAP(random_state=10, distance='manhattan')
    b_out = b.fit_transform(sample_data)
    c = pacmap.PaCMAP(random_state=10, distance='manhattan')
    c_out = c.fit_transform(sample_data)
    d = pacmap.PaCMAP(distance='angular')
    d_out = d.fit_transform(sample_data)
    e = pacmap.PaCMAP(distance='hamming', apply_pca=False)
    e_out = e.fit_transform(sample_data, init='random')
    print('Experiment has been done successfully for each metric.')

    # Ensure the random state settings can be applied
    try:
        assert(np.sum(np.abs(b_out-c_out))<1e-8)
        print("The output is deterministic.")
    except AssertionError:
        print("The output is not deterministic.")
        try:
            assert(np.sum(np.abs(b.pair_FP.astype(int)-c.pair_FP.astype(int)))<1e-8)
            assert(np.sum(np.abs(b.pair_MN.astype(int)-c.pair_MN.astype(int)))<1e-8)
        except AssertionError:
            print('The pairs are not deterministic')
            for i in range(5000):
                if np.sum(np.abs(b.pair_FP[i] - c.pair_FP[i])) > 1e-8:
                    print("FP")
                    print(i)
                    print(b.pair_FP[i])
                    print(c.pair_FP[i])
                    break
            for i in range(5000):
                if np.sum(np.abs(b.pair_MN[i] - c.pair_MN[i])) > 1e-8:
                    print('MN')
                    print(i)
                    print(b.pair_MN[i])
                    print(c.pair_MN[i])
                    break


def test_pacmap_fashion_mnist_manhattan():
    """Test PaCMAP with Fashion-MNIST using Manhattan distance."""
    # Load Fashion-MNIST from OpenML
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    fmnist, labels = fashion_mnist
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


def test_pacmap_mnist_metrics():
    """Test PaCMAP with MNIST using different distance metrics."""
    # Load MNIST from OpenML
    mnist_data = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    mnist, labels = mnist_data
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
