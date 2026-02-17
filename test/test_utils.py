import os
import pytest 
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path



def get_available_backends():
    backends = ['annoy']
    try: import faiss; backends.append('faiss')
    except ImportError: pass
    try: import voyager; backends.append('voyager')
    except ImportError: pass
    return backends

def get_backend_metric_pairs():
    """Returns a list of valid (backend, metric) tuples based on library support."""
    available = get_available_backends()
    pairs = []
    for b in available:
        pairs.append((b, 'euclidean'))
        pairs.append((b, 'angular'))
        if b in ['annoy', 'faiss']:
            pairs.append((b, 'manhattan'))
        if b == 'annoy':
            pairs.append((b, 'hamming'))
    return pairs


@pytest.fixture
def sample_data():
    """Standard dataset fixture (1000 samples, 20 features)."""
    return np.random.normal(size=(1000, 20))

@pytest.fixture
def mnist_data(openml_datasets):
    """Load MNIST data for testing"""
    print("Loading data")
    # Load MNIST from fixture
    mnist, labels = openml_datasets["mnist_784"]
    mnist = mnist.reshape(mnist.shape[0], -1)

    # Use subset for faster testing
    mnist = mnist[:5000]
    labels = labels[:5000].astype(int)
    return mnist, labels


@pytest.fixture
def mnist_data(openml_datasets):
    """Load MNIST data for testing"""
    print("Loading data")
    # Load MNIST from fixture
    mnist, labels = openml_datasets["mnist_784"]
    mnist = mnist.reshape(mnist.shape[0], -1)

    # Use subset for faster testing
    mnist = mnist[:5000]
    labels = labels[:5000].astype(int)
    return mnist, labels


@pytest.fixture
def fmnist_data(openml_datasets):
    """Load Fashion-MNIST data for testing"""
    print("Loading Fashion-MNIST data")
    # Load Fashion-MNIST from fixture
    fmnist, labels = openml_datasets["Fashion-MNIST"]
    fmnist = fmnist.reshape(fmnist.shape[0], -1)

    # Use subset for faster testing
    fmnist = fmnist[:5000]
    labels = labels[:5000].astype(int)
    return fmnist, labels

@pytest.fixture
def iris_data():
    """Load iris dataset for testing."""
    from sklearn import datasets
    iris = datasets.load_iris()
    return iris["data"], iris["target"]

@pytest.fixture
def output_dir():
    """Ensures the output directory exists and returns the path."""
    path = "./test/output"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def generate_figure(embedding, labels, title):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, c=labels, cmap='Spectral')
    ax.axis('off')
    ax.set_title(title)
    plt.savefig(f"./test/output/{title}.png")
    plt.close()


def generate_combined_figure(embeddings, labels, titles, theme_title):
    len_subfigs = len(embeddings)
    assert len(labels) == len_subfigs
    assert len(titles) == len_subfigs
    n_rows = (len_subfigs + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize = (18, n_rows * 6))
    axes = axes.flatten()
    for i in range(len_subfigs):
        ax = axes[i]
        embedding = embeddings[i]
        label = labels[i]
        title = titles[i]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, c=label, cmap='Spectral')
        ax.axis('off')
        ax.set_title(title)
    for i in range(3 * n_rows - len_subfigs):
        axes[-i - 1].axis('off')
    plt.savefig(f"./test/output/{theme_title}.png")
    plt.close()

