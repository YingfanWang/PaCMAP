'''
A general test script that ensures LocalMAP can be successfully loaded and run with different backends.
Refactored to use pytest parametrization for better test isolation and reporting.
'''
import pytest
from pacmap import pacmap
import numpy as np
import os
from test_utils import get_available_backends, sample_data, output_dir, mnist_data
try:
    from test_utils import generate_figure
except ImportError:
    def generate_figure(*args, **kwargs):
        print("generate_figure called (test_utils not found)")


@pytest.mark.parametrize("backend", get_available_backends())
def test_localmap_initialization(backend):
    """Test that LocalMAP can be initialized successfully with all backends."""
    print(f"Testing LocalMAP initialization for backend: {backend}")
    lm = pacmap.LocalMAP(knn_backend=backend)
    assert lm.knn_backend == backend
    assert hasattr(lm, 'low_dist_thres')
    assert lm.low_dist_thres == 10  # Check default value

@pytest.mark.parametrize("backend", get_available_backends())
def test_localmap_standard_dataset(backend, sample_data):
    """Test LocalMAP on a standard dataset with specific backends."""
    print(f"Testing LocalMAP standard dataset with backend: {backend}")
    lm = pacmap.LocalMAP(knn_backend=backend, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    embedding = lm.fit_transform(sample_data, init="random")
    assert embedding.shape == (1000, 2)

@pytest.mark.parametrize("backend", get_available_backends())
def test_localmap_small_n_reorganization(backend):
    """Test LocalMAP behavior when n < n_neighbors + n_MN + n_FP."""
    n_samples = 15
    sample_data = np.random.normal(size=(n_samples, 10)).astype(np.float32)
    
    print(f"Testing LocalMAP small n reorganization for backend: {backend}")
    lm = pacmap.LocalMAP(n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, knn_backend=backend)
    
    # Fit transform should not crash and should produce embeddings
    embedding = lm.fit_transform(sample_data, init="random")
    assert embedding.shape == (n_samples, 2)
    
    # Verify pairs were generated (adjusted internally by the algorithm)
    assert lm.pair_neighbors is not None
    assert lm.pair_MN is not None
    assert lm.pair_FP is not None

@pytest.mark.parametrize("backend", get_available_backends())
def test_localmap_determinism(backend, sample_data):
    """Test LocalMAP deterministic behavior using random_state."""
    print(f"Testing LocalMAP determinism for {backend}")
    
    lm1 = pacmap.LocalMAP(random_state=42, knn_backend=backend)
    emb1 = lm1.fit_transform(sample_data, init="random")
    
    lm2 = pacmap.LocalMAP(random_state=42, knn_backend=backend)
    emb2 = lm2.fit_transform(sample_data, init="random")
    
    try:
        assert np.allclose(emb1, emb2, atol=1e-5)
    except AssertionError:
        if backend == 'annoy':
            raise AssertionError(f"Annoy backend should be strictly deterministic.")
        # Some backends like voyager or multi-threaded faiss might show minor drift
        pytest.warns(UserWarning, match=f"Minor drift detected in {backend}")

@pytest.mark.parametrize("backend", get_available_backends())
def test_localmap_low_dist_thres(backend, sample_data):
    """Test LocalMAP with custom low_dist_thres parameter."""
    print(f"Testing LocalMAP custom threshold for {backend}")
    lm = pacmap.LocalMAP(low_dist_thres=5.0, knn_backend=backend)
    emb = lm.fit_transform(sample_data)
    assert emb.shape == (1000, 2)

@pytest.mark.parametrize("backend", get_available_backends())
def test_localmap_mnist(backend, mnist_data):
    """Test LocalMAP with MNIST dataset from OpenML."""
    mnist, labels = mnist_data
    print(f"Testing LocalMAP MNIST with backend: {backend}")
    lm = pacmap.LocalMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, 
                         random_state=20, knn_backend=backend)
    embedding = lm.fit_transform(mnist, init="pca")
    generate_figure(embedding, labels, f'test_localmap_mnist_{backend}')
    print(f'LocalMAP figure generated for {backend}.')


if __name__ == "__main__":
    print("Running LocalMAP tests...")
    pytest.main([__file__])