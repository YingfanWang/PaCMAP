'''
A general test script that ensures PaCMAP can be successfully loaded and run with different backends.
Refactored to use pytest parametrization for better test isolation and reporting.
'''
import pytest
from pacmap import pacmap
import numpy as np
import matplotlib.pyplot as plt
from test_utils import get_available_backends, sample_data, output_dir, mnist_data, fmnist_data
from pathlib import Path

try:
    from test_utils import generate_figure
except ImportError:
    def generate_figure(*args, **kwargs):
        print("generate_figure called (test_utils not found)")
        
def test_pacmap_initialization_default():
    """Test that PaCMAP can be initialized with default settings."""
    pm = pacmap.PaCMAP()
    assert pm is not None

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_initialization_backends(backend):
    """Test initialization for each specific backend."""
    pm = pacmap.PaCMAP(knn_backend=backend)
    assert pm.knn_backend == backend

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_standard_dataset(backend, sample_data):
    """Test PaCMAP on a standard dataset with all available backends."""
    print(f"Testing standard dataset with backend: {backend}")
    pm = pacmap.PaCMAP(knn_backend=backend)
    out = pm.fit_transform(sample_data)
    assert out.shape == (1000, 2)

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_deterministic_random_state(backend, sample_data):
    """Test PaCMAP deterministic behavior."""
    # Voyager might require specific handling if it uses internal threading seeds
    print(f"Testing deterministic behavior for backend: {backend}")
    b = pacmap.PaCMAP(random_state=10, knn_backend=backend)
    b_out = b.fit_transform(sample_data)
    
    c = pacmap.PaCMAP(random_state=10, knn_backend=backend)
    c_out = c.fit_transform(sample_data)

    try:
        assert np.allclose(b_out, c_out, atol=1e-8)
    except AssertionError:
        debug_nondeterminism(b, c)
        raise AssertionError(f"Backend {backend} failed deterministic test")

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_nondeterministic_default(backend, sample_data):
    """Test PaCMAP non-deterministic behavior with default settings."""
    d = pacmap.PaCMAP(knn_backend=backend)
    d_out = d.fit_transform(sample_data)
    
    e = pacmap.PaCMAP(knn_backend=backend)
    e_out = e.fit_transform(sample_data)

    assert not np.allclose(d_out, e_out, atol=1e-8)

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_small_dataset(backend):
    """Test PaCMAP with small dataset where dimensions > samples."""
    sample_small_data = np.random.normal(size=(40, 60))
    f = pacmap.PaCMAP(random_state=20, knn_backend=backend)
    f_out = f.fit_transform(sample_small_data)
    assert f_out.shape == (40, 2)

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_large_dataset(backend):
    """Test PaCMAP with larger dataset."""
    sample_large_data = np.random.normal(size=(2000, 150))
    g = pacmap.PaCMAP(knn_backend=backend)
    g_out = g.fit_transform(sample_large_data)
    assert g_out.shape == (2000, 2)

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_same_dimensional_2d(backend):
    """Test PaCMAP with same dimensional data (2D -> 2D)."""
    data = np.random.normal(size=(1000, 2))
    pm = pacmap.PaCMAP(knn_backend=backend)
    out = pm.fit_transform(data)
    assert out.shape == (1000, 2)

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_same_dimensional_3d(backend):
    """Test PaCMAP with same dimensional data (3D -> 3D)."""
    data = np.random.normal(size=(1000, 3))
    pm = pacmap.PaCMAP(n_components=3, knn_backend=backend)
    out = pm.fit_transform(data)
    assert out.shape == (1000, 3)

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_fashion_mnist(backend, fmnist_data):
    """Test PaCMAP with Fashion-MNIST subset."""
    fmnist, labels = fmnist_data
    fmnist = fmnist[:500].reshape(500, -1)
    labels = labels[:500].astype(int)

    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, knn_backend=backend)
    embedding = reducer.fit_transform(fmnist, init="pca")
    generate_figure(embedding, labels, f'test_fmnist_seed_{backend}')

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_mnist(tmp_path, backend, output_dir, mnist_data):
    """Test PaCMAP with MNIST dataset and save/load."""
    mnist, labels = mnist_data
    mnist = mnist[:500].reshape(500, -1)
    labels = labels[:500].astype(int)
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, knn_backend=backend)
    embedding = reducer.fit_transform(mnist, init="pca")
    generate_figure(embedding, labels, f'test_mnist_seed_{backend}')
    plt.savefig(f"{output_dir}/test_mnist_seed_{backend}.png")

    # Test Save/Load
    reducer_to_save = pacmap.PaCMAP(n_components=2, n_neighbors=10, save_tree=True, knn_backend=backend)
    reducer_to_save.fit_transform(mnist, init="pca")
    
    save_path = tmp_path / f"mnist_reducer_{backend}"
    pacmap.save(reducer_to_save, str(save_path))
    
    reducer_loaded = pacmap.load(str(save_path))
    assert reducer_loaded.knn_backend == backend
    
    embedding_loaded = reducer_loaded.transform(mnist)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(embedding_loaded[:, 0], embedding_loaded[:, 1], s=0.5, c=labels, cmap='Spectral')
    ax.axis('off')
    ax.set_title(f'test_saveload_{backend}')
    plt.savefig(f"{output_dir}/test_saveload_after_{backend}.png")


def debug_nondeterminism(b, c):
    print("The output is not deterministic. Checking pairs...")
    if hasattr(b, 'pair_FP') and hasattr(c, 'pair_FP'):
        fp_diff = np.sum(np.abs(b.pair_FP.astype(int) - c.pair_FP.astype(int)))
        mn_diff = np.sum(np.abs(b.pair_MN.astype(int) - c.pair_MN.astype(int)))
        if fp_diff > 1e-8 or mn_diff > 1e-8:
            print(f"Pairs are not deterministic. FP diff: {fp_diff}, MN diff: {mn_diff}")

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_save(tmp_path, backend, sample_data):
    """Test that PaCMAP instances are saved correctly with their respective backend trees."""
    print(f"Testing save functionality for backend: {backend}")
    
    # Initialize and fit PaCMAP with save_tree=True
    pm = pacmap.PaCMAP(n_components=2, save_tree=True, knn_backend=backend)
    pm.fit(sample_data)
    
    # Define save path prefix using pytest's tmp_path fixture
    save_prefix = tmp_path / f"test_save_{backend}"
    
    # Call the save function
    pacmap.save(pm, str(save_prefix))
    
    # 1. Assert the pickle file was created
    pkl_path = save_prefix.with_suffix(".pkl")
    assert pkl_path.exists(), f"Pickle file was not created for {backend}"
    assert pkl_path.stat().st_size > 0, f"Pickle file is empty for {backend}"
    
    # 2. Assert the backend-specific tree index file was created
    ext_map = {
        'annoy': '.ann',
        'faiss': '.faiss',
        'voyager': '.voyager'
    }
    
    tree_ext = ext_map.get(backend, '.ann') # Fallback to .ann as default
    tree_path = save_prefix.with_suffix(tree_ext)
    
    assert tree_path.exists(), f"Tree index file ({tree_ext}) was not created for {backend}"
    assert tree_path.stat().st_size > 0, f"Tree index file is empty for {backend}"

if __name__ == "__main__":
    print("Starting PaCMAP backend tests via pytest...")
    pytest.main([__file__])