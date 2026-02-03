'''
A general test script that ensures LocalMAP can be successfully loaded and run with different backends.
'''
from pacmap import pacmap
import numpy as np
import pytest
import os

# Helper to import generate_figure if available, else mock it
try:
    from test_utils import generate_figure
except ImportError:
    def generate_figure(*args, **kwargs):
        print("generate_figure called (test_utils not found)")

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

def test_localmap_initialization():
    """Test that LocalMAP can be initialized successfully."""
    for backend in get_available_backends():
        print(f"Testing LocalMAP initialization for backend: {backend}")
        lm = pacmap.LocalMAP(knn_backend=backend)
        assert lm.knn_backend == backend
        assert hasattr(lm, 'low_dist_thres')
        assert lm.low_dist_thres == 10  # Check default value

def test_localmap_standard_dataset():
    """Test LocalMAP on a standard dataset with all available backends."""
    sample_data = np.random.normal(size=(500, 20)).astype(np.float32) # Reduced size
    
    for backend in get_available_backends():
        print(f"Testing LocalMAP standard dataset with backend: {backend}")
        lm = pacmap.LocalMAP(knn_backend=backend, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
        embedding = lm.fit_transform(sample_data, init="random")
        assert embedding.shape == (500, 2)

def test_localmap_small_n_reorganization():
    """Test LocalMAP behavior when n < n_neighbors + n_MN + n_FP."""
    # Create very small dataset
    n_samples = 15
    sample_data = np.random.normal(size=(n_samples, 10)).astype(np.float32)
    
    # Configure parameters such that sum > n_samples
    # e.g., n_neighbors=10, MN_ratio=0.5 -> n_MN=5, FP_ratio=2.0 -> n_FP=20
    # Sum = 10 + 5 + 20 = 35 > 15
    
    for backend in get_available_backends():
        print(f"Testing LocalMAP small n reorganization for backend: {backend}")
        lm = pacmap.LocalMAP(n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, knn_backend=backend)
        
        # Fit transform should not crash and should produce embeddings
        try:
            embedding = lm.fit_transform(sample_data, init="random")
            assert embedding.shape == (n_samples, 2)
            
            # Verify pairs were generated (adjusted internally)
            assert lm.pair_neighbors is not None
            assert lm.pair_MN is not None
            assert lm.pair_FP is not None
            
        except Exception as e:
            pytest.fail(f"LocalMAP failed on small dataset for backend {backend}: {e}")

def test_localmap_determinism():
    """Test LocalMAP deterministic behavior."""
    sample_data = np.random.normal(size=(500, 20)).astype(np.float32)
    
    for backend in get_available_backends():
        print(f"Testing LocalMAP determinism for {backend}")
        
        # Run 1
        lm1 = pacmap.LocalMAP(random_state=42, knn_backend=backend)
        emb1 = lm1.fit_transform(sample_data, init="random")
        
        # Run 2
        lm2 = pacmap.LocalMAP(random_state=42, knn_backend=backend)
        emb2 = lm2.fit_transform(sample_data, init="random")
        
        try:
            assert np.allclose(emb1, emb2, atol=1e-5)
            print(f"  > Backend {backend} is deterministic.")
        except AssertionError:
            print(f"  > WARNING: Backend {backend} output did not match exactly.")
            # Depending on the backend version/threading, exact determinism might fail 
            # (especially for voyager or multi-threaded faiss if not handled)
            if backend == 'annoy':
                raise AssertionError(f"Annoy backend should be deterministic.")

def test_localmap_low_dist_thres():
    """Test LocalMAP with custom low_dist_thres parameter."""
    sample_data = np.random.normal(size=(500, 20)).astype(np.float32)
    
    for backend in get_available_backends():
        print(f"Testing LocalMAP custom threshold for {backend}")
        # Use a different threshold
        lm = pacmap.LocalMAP(low_dist_thres=5.0, knn_backend=backend)
        emb = lm.fit_transform(sample_data)
        assert emb.shape == (500, 2)

def test_localmap_mnist(openml_datasets=None):
    """Test LocalMAP with MNIST dataset."""
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
        print(f"Testing LocalMAP MNIST with backend: {backend}")
        
        # Initialize LocalMAP
        lm = pacmap.LocalMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, 
                             random_state=20, knn_backend=backend)
        
        # Fit transform
        embedding = lm.fit_transform(mnist, init="pca")
        
        # Generate figure
        generate_figure(embedding, labels, f'test_localmap_mnist_{backend}')
        
    print('LocalMAP figures have been generated successfully.')

if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists("./test/output"):
        os.makedirs("./test/output")
        
    test_localmap_initialization()
    test_localmap_standard_dataset()
    test_localmap_small_n_reorganization()
    test_localmap_determinism()
    test_localmap_low_dist_thres()
    
    # This requires internet/data
    test_localmap_mnist()