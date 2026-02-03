'''
A general test script that ensures PaCMAP can be successfully loaded and run with different backends.
'''
from pacmap import pacmap
import numpy as np
import matplotlib.pyplot as plt
import test_utils
import os

# Helper to get available backends
def get_available_backends():
    backends = ['annoy']
    try:
        import faiss
        backends.append('faiss')
    except ImportError:
        pass
    try:
        import voyager
        backends.append('voyager')
    except ImportError:
        pass
    return backends

def test_pacmap_initialization():
    """Test that PaCMAP can be initialized successfully."""
    # Try initialize default
    pacmap.PaCMAP()
    print(pacmap.PaCMAP())
    
    # Try initialize specific backends if available
    for backend in get_available_backends():
        pm = pacmap.PaCMAP(knn_backend=backend)
        assert pm.knn_backend == backend

def test_pacmap_standard_dataset():
    """Test PaCMAP on a standard dataset with all available backends."""
    sample_data = np.random.normal(size=(1000, 20)) # Reduced size
    
    for backend in get_available_backends():
        print(f"Testing standard dataset with backend: {backend}")
        a = pacmap.PaCMAP(knn_backend=backend)
        a_out = a.fit_transform(sample_data)
        assert a_out.shape == (1000, 2)

def test_pacmap_deterministic_random_state():
    """Test PaCMAP deterministic behavior."""
    # Initialize sample data
    sample_data = np.random.normal(size=(1000, 20))
    
    # Test for each backend
    for backend in get_available_backends():
        if backend == "voyager":
            # Faiss uses a fixed seed internally for reproducibility.
            print("Skipping non-deterministic test for voyager backend.")
            continue
        print(f"Testing deterministic behavior for backend: {backend}")
        b = pacmap.PaCMAP(random_state=10, knn_backend=backend)
        b_out = b.fit_transform(sample_data)
        c = pacmap.PaCMAP(random_state=10, knn_backend=backend)
        c_out = c.fit_transform(sample_data)

        try:
            assert np.allclose(b_out, c_out, atol=1e-8)
        except AssertionError:
            # Print debug output and re-raise error.
            debug_nondeterminism(b, c)
            raise AssertionError(f"Backend {backend} failed deterministic test")


def test_pacmap_nondeterministic_default():
    """Test PaCMAP non-deterministic behavior with default basic functionality."""
    sample_data = np.random.normal(size=(1000, 20))
    
    for backend in get_available_backends():
        d = pacmap.PaCMAP(knn_backend=backend)
        d_out = d.fit_transform(sample_data)
        e = pacmap.PaCMAP(knn_backend=backend)
        e_out = e.fit_transform(sample_data)

        assert not np.allclose(d_out, e_out, atol=1e-8)


def test_pacmap_small_dataset():
    """Test PaCMAP with small dataset where dimensions > samples."""
    sample_small_data = np.random.normal(size=(40, 60))  # dim > size
    for backend in get_available_backends():
        f = pacmap.PaCMAP(random_state=20, knn_backend=backend)
        f_out = f.fit_transform(sample_small_data)


def test_pacmap_large_dataset():
    """Test PaCMAP with larger dataset."""
    sample_large_data = np.random.normal(size=(2000, 150)) # Reduced from 10000 for quick testing of all backends
    for backend in get_available_backends():
        g = pacmap.PaCMAP(knn_backend=backend)
        g_out = g.fit_transform(sample_large_data)


def test_pacmap_same_dimensional_2d():
    """Test PaCMAP with same dimensional data (2D -> 2D)."""
    same_dimensional_data = np.random.normal(size=(1000, 2))
    for backend in get_available_backends():
        h = pacmap.PaCMAP(knn_backend=backend)
        h_out = h.fit_transform(same_dimensional_data)


def test_pacmap_same_dimensional_3d():
    """Test PaCMAP with same dimensional data (3D -> 3D)."""
    three_dimensional_data = np.random.normal(size=(1000, 3))
    for backend in get_available_backends():
        h = pacmap.PaCMAP(n_components=3, knn_backend=backend)
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

def test_pacmap_fashion_mnist(openml_datasets=None):
    """Test PaCMAP with Fashion-MNIST dataset from OpenML."""
    if openml_datasets is None:
        try:
            from sklearn.datasets import fetch_openml
            fmnist, labels = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
        except Exception as e:
            print(f"Skipping Fashion-MNIST test due to download failure: {e}")
            return
    else:
        fmnist, labels = openml_datasets["Fashion-MNIST"]
        fmnist = fmnist.reshape(fmnist.shape[0], -1)

    # Use subset for faster testing
    fmnist = fmnist[:500]
    labels = labels[:500].astype(int)

    for backend in get_available_backends():
        print(f"Testing Fashion-MNIST with backend: {backend}")
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, knn_backend=backend)
        embedding = reducer.fit_transform(fmnist, init="pca")
        test_utils.generate_figure(embedding, labels, f'test_fmnist_seed_{backend}')


def test_pacmap_mnist(tmp_path, openml_datasets=None):
    """Test PaCMAP with MNIST dataset from OpenML."""
    if openml_datasets is None:
        try:
            from sklearn.datasets import fetch_openml
            mnist, labels = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        except Exception as e:
            print(f"Skipping MNIST test due to download failure: {e}")
            return
    else:
        mnist, labels = openml_datasets["mnist_784"]
        mnist = mnist.reshape(mnist.shape[0], -1)

    # Use subset for faster testing
    mnist = mnist[:500]
    labels = labels[:500].astype(int)

    for backend in get_available_backends():
        print(f"Testing MNIST with backend: {backend}")
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, knn_backend=backend)
        embedding = reducer.fit_transform(mnist, init="pca")
        test_utils.generate_figure(embedding, labels, f'test_mnist_seed_{backend}')
        
        output_dir = "./test/output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(f"{output_dir}/test_mnist_seed_{backend}.png")

        # Test Save/Load
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, save_tree=True, knn_backend=backend)
        embedding = reducer.fit_transform(mnist, init="pca")
        
        # Save and load
        save_path = tmp_path / f"mnist_reducer_{backend}"
        pacmap.save(reducer, str(save_path))
        
        reducer_loaded = pacmap.load(str(save_path))
        # Verify backend persisted
        assert reducer_loaded.knn_backend == backend
        
        embedding_loaded = reducer_loaded.transform(mnist)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(embedding_loaded[:, 0], embedding_loaded[:, 1], s=0.5, c=labels, cmap='Spectral')
        ax.axis('off')
        ax.set_title(f'test_saveload_{backend}')
        plt.savefig(f"{output_dir}/test_saveload_after_{backend}.png")

    print("Figures have been generated successfully.")

# TODO: Remove this is we want to commit to using pytest.
if __name__ == "__main__":
    # Backward compatibility - can still run as script
    import tempfile
    from pathlib import Path
    
    # Ensure output directory exists
    if not os.path.exists("./test/output"):
        os.makedirs("./test/output")

    test_pacmap_initialization()
    test_pacmap_standard_dataset()
    test_pacmap_deterministic_random_state()
    test_pacmap_nondeterministic_default()
    test_pacmap_small_dataset()
    test_pacmap_large_dataset()
    test_pacmap_same_dimensional_2d()
    test_pacmap_same_dimensional_3d()
    
    # These might be slow without the mock/fixture data
    print("Running large dataset tests (FMNIST, MNIST)...")
    test_pacmap_fashion_mnist()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_pacmap_mnist(Path(tmp_dir))