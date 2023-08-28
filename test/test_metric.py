'''
A test script that ensures PaCMAP can be successfully used with other metrics.
'''

import pacmap
import numpy as np
import matplotlib.pyplot as plt
from test_utils import *

if __name__ == "__main__":
    # Try initialize
    pacmap.PaCMAP(distance='manhattan')
    pacmap.PaCMAP(distance='angular')
    pacmap.PaCMAP(distance='hamming')
    print("Instance initialized successfully.")
    try:
        pacmap.PaCMAP(distance='unknown')
    except NotImplementedError:
        print("Not implemented error raised successfully")

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

    # FMNIST
    fmnist = np.load("/Users/hyhuang/Desktop/MNIST/fmnist_images.npy", allow_pickle=True)
    fmnist = fmnist.reshape(fmnist.shape[0], -1)
    labels = np.load("/Users/hyhuang/Desktop/MNIST/fmnist_labels.npy", allow_pickle=True)
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, distance='manhattan')
    embedding = reducer.fit_transform(fmnist)
    generate_figure(embedding, labels, 'test_fmnist_manhattan')

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, distance='manhattan')
    embedding = reducer.fit_transform(fmnist)
    generate_figure(embedding, labels, 'test_fmnist_manhattan_noseed')

    # MNIST
    mnist = np.load("/Users/hyhuang/Desktop/MNIST/mnist_images.npy", allow_pickle=True)
    mnist = mnist.reshape(mnist.shape[0], -1)
    labels = np.load("/Users/hyhuang/Desktop/MNIST/mnist_labels.npy", allow_pickle=True)
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
