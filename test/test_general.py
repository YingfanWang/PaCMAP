'''
A general test script that ensures PaCMAP can be successfully loaded.
'''
from pacmap import pacmap
import numpy as np
import matplotlib.pyplot as plt
import test_utils


def test_pacmap_initialization():
    """Test that PaCMAP can be initialized successfully."""
    # Try initialize
    pacmap.PaCMAP()
    # print instance
    print(pacmap.PaCMAP())

def test_pacmap_standard_dataset():
    """Test PaCMAP on a standard dataset."""
    sample_data = np.random.normal(size=(10000, 20))
    a = pacmap.PaCMAP()
    a_out = a.fit_transform(sample_data)

def test_pacmap_deterministic_random_state():
    """Test PaCMAP deterministic behavior."""
    # Initialize sample data
    sample_data = np.random.normal(size=(10000, 20))
    b = pacmap.PaCMAP(random_state=10)
    b_out = b.fit_transform(sample_data)
    c = pacmap.PaCMAP(random_state=10)
    c_out = c.fit_transform(sample_data)

    try:
        assert np.allclose(b_out, c_out, atol=1e-8)
    except AssertionError:
        # Print debug output and re-raise error.
        debug_nondeterminism(b, c)
        raise AssertionError


def test_pacmap_nondeterministic_default():
    """Test PaCMAP non-deterministic behavior with default basic functionality."""
    sample_data = np.random.normal(size=(10000, 20))
    d = pacmap.PaCMAP()
    d_out = d.fit_transform(sample_data)
    e = pacmap.PaCMAP()
    e_out = e.fit_transform(sample_data)

    assert not np.allclose(d_out, e_out, atol=1e-8)


def test_pacmap_small_dataset():
    """Test PaCMAP with small dataset where dimensions > samples."""
    sample_small_data = np.random.normal(size=(40, 60))  # dim > size
    f = pacmap.PaCMAP(random_state=20)
    f_out = f.fit_transform(sample_small_data)


def test_pacmap_large_dataset():
    """Test PaCMAP with larger dataset."""
    sample_large_data = np.random.normal(size=(10000, 150))
    g = pacmap.PaCMAP()
    g_out = g.fit_transform(sample_large_data)


def test_pacmap_same_dimensional_2d():
    """Test PaCMAP with same dimensional data (2D -> 2D)."""
    same_dimensional_data = np.random.normal(size=(10000, 2))
    h = pacmap.PaCMAP()
    h_out = h.fit_transform(same_dimensional_data)


def test_pacmap_same_dimensional_3d():
    """Test PaCMAP with same dimensional data (3D -> 3D)."""
    three_dimensional_data = np.random.normal(size=(10000, 3))
    h = pacmap.PaCMAP(n_components=3)
    h_out = h.fit_transform(three_dimensional_data)

def debug_nondeterminism(b, c):
    DEBUG = True
    if DEBUG:
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

def test_pacmap_fashion_mnist(openml_datasets):
    """Test PaCMAP with Fashion-MNIST dataset from OpenML."""
    # Load Fashion-MNIST from fixture
    fmnist, labels = openml_datasets["Fashion-MNIST"]
    fmnist = fmnist.reshape(fmnist.shape[0], -1)

    # Use subset for faster testing
    fmnist = fmnist[:1000]
    labels = labels[:1000].astype(int)

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20)
    embedding = reducer.fit_transform(fmnist, init="pca")
    test_utils.generate_figure(embedding, labels, 'test_fmnist_seed')

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    embedding = reducer.fit_transform(fmnist, init="pca")
    test_utils.generate_figure(embedding, labels, 'test_fmnist_noseed')


def test_pacmap_mnist(tmp_path, openml_datasets):
    """Test PaCMAP with MNIST dataset from OpenML."""
    # Load MNIST from fixture
    mnist, labels = openml_datasets["mnist_784"]
    mnist = mnist.reshape(mnist.shape[0], -1)

    # Use subset for faster testing
    mnist = mnist[:1000]
    labels = labels[:1000].astype(int)

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20)
    embedding = reducer.fit_transform(mnist, init="pca")
    test_utils.generate_figure(embedding, labels, 'test_mnist_seed')

    plt.savefig("./test/output/test_mnist_seed.png")

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, save_tree=True)
    embedding = reducer.fit_transform(mnist, init="pca")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, c=labels, cmap='Spectral')
    ax.axis('off')
    ax.set_title('test_mnist_noseed')
    plt.savefig("./test/output/test_mnist_noseed.png")

    # Save and load
    save_path = tmp_path / "mnist_reducer"
    pacmap.save(reducer, str(save_path))
    embedding = reducer.transform(mnist)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, c=labels, cmap='Spectral')
    ax.axis('off')
    ax.set_title('test_saveload')
    plt.savefig("./test/output/test_saveload_before.png")

    reducer = pacmap.load(str(save_path))
    embedding = reducer.transform(mnist)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, c=labels, cmap='Spectral')
    ax.axis('off')
    ax.set_title('test_saveload')
    plt.savefig("./test/output/test_saveload_after.png")

    print("Figures have been generated successfully.")

# TODO: Remove this is we want to commit to using pytest.
if __name__ == "__main__":
    # Backward compatibility - can still run as script
    import tempfile

    test_pacmap_initialization()
    test_pacmap_standard_dataset()
    test_pacmap_deterministic_random_state()
    test_pacmap_nondeterministic_default()
    test_pacmap_small_dataset()
    test_pacmap_large_dataset()
    test_pacmap_same_dimensional_2d()
    test_pacmap_same_dimensional_3d()
    test_pacmap_fashion_mnist()
    with tempfile.TemporaryDirectory() as tmp_dir:
        from pathlib import Path

        test_pacmap_mnist(Path(tmp_dir))
