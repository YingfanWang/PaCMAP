'''
A test script that ensures PaCMAP can be successfully used with other metrics across different backends.
Refactored for pytest with optimized metric-backend mapping.
'''
import pytest
from pacmap import pacmap
import numpy as np
from test_utils import get_available_backends, get_backend_metric_pairs, sample_data, output_dir

try:
    from test_utils import generate_figure
except ImportError:
    def generate_figure(*args, **kwargs):
        print("generate_figure called (test_utils not found)")


@pytest.mark.parametrize("backend, metric", get_backend_metric_pairs())
def test_pacmap_metric_initialization(backend, metric):
    """Test that PaCMAP initializes correctly for all supported backend/metric pairs."""
    pm = pacmap.PaCMAP(distance=metric, knn_backend=backend)
    assert pm.distance == metric
    assert pm.knn_backend == backend

def test_pacmap_metric_unknown_fails():
    """Test that PaCMAP fails to initialize with unknown distance metrics."""
    with pytest.raises(NotImplementedError):
        pacmap.PaCMAP(distance='unknown')

@pytest.mark.parametrize("backend", [b for b in get_available_backends() if b != 'voyager'])
def test_pacmap_metric_deterministic_manhattan(backend, sample_data):
    """Test PaCMAP deterministic behavior with Manhattan distance."""
    b = pacmap.PaCMAP(random_state=10, distance='manhattan', knn_backend=backend)
    b_out = b.fit_transform(sample_data)
    
    c = pacmap.PaCMAP(random_state=10, distance='manhattan', knn_backend=backend)
    c_out = c.fit_transform(sample_data)
    
    try:
        assert np.allclose(b_out, c_out, atol=1e-8)
    except AssertionError:
        debug_nondeterminism(b, c)
        raise AssertionError(f"Backend {backend} failed deterministic Manhattan test")

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_metric_angular(backend, sample_data):
    """Test PaCMAP works with angular distance metric."""
    pm = pacmap.PaCMAP(distance='angular', knn_backend=backend)
    out = pm.fit_transform(sample_data)
    assert out.shape == (1000, 2)

@pytest.mark.parametrize("backend", [b for b in get_available_backends() if b == 'annoy'])
def test_pacmap_metric_hamming(backend, sample_data):
    """Test PaCMAP works with hamming distance (Annoy only)."""
    pm = pacmap.PaCMAP(distance='hamming', apply_pca=False, knn_backend=backend)
    out = pm.fit_transform(sample_data, init='random')
    assert out.shape == (1000, 2)

@pytest.mark.parametrize("backend", [b for b in get_available_backends() if b != 'voyager'])
def test_pacmap_fashion_mnist_manhattan(backend):
    """Test PaCMAP with Fashion-MNIST using Manhattan distance."""
    try:
        from sklearn.datasets import fetch_openml
        fmnist, labels = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    except Exception as e:
        pytest.skip(f"Skipping FMNIST: {e}")

    fmnist = fmnist[:500].reshape(500, -1)
    labels = labels[:500].astype(int)
    
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=20, 
                            distance='manhattan', knn_backend=backend)
    embedding = reducer.fit_transform(fmnist)
    generate_figure(embedding, labels, f'test_fmnist_manhattan_{backend}')

@pytest.mark.parametrize("backend, metric", get_backend_metric_pairs())
def test_pacmap_mnist_metrics(backend, metric):
    """Comprehensive MNIST test for all valid backend/metric combinations."""
    try:
        from sklearn.datasets import fetch_openml
        mnist, labels = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    except Exception as e:
        pytest.skip(f"Skipping MNIST: {e}")

    mnist = mnist[:500].reshape(500, -1)
    labels = labels[:500].astype(int)
    
    apply_pca = False if metric == 'hamming' else True
    
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, distance=metric, 
                            apply_pca=apply_pca, knn_backend=backend, random_state=42)
    embedding = reducer.fit_transform(mnist)
    generate_figure(embedding, labels, f'test_mnist_{metric}_{backend}')

def debug_nondeterminism(b, c):
    print("Checking for pair mismatches...")
    for attr in ['pair_FP', 'pair_MN']:
        val_b = getattr(b, attr, None)
        val_c = getattr(c, attr, None)
        if val_b is not None and val_c is not None:
            if not np.array_equal(val_b, val_c):
                print(f"Mismatch found in {attr}")


if __name__ == "__main__":
    pytest.main([__file__])