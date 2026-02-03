'''
A test script that ensures PaCMAP can be successfully used with other metrics across different backends.
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

def debug_nondeterminism(b, c):
    DEBUG = True
    if DEBUG:
        print("The output is not deterministic.")
        try:
            assert(np.sum(np.abs(b.pair_FP.astype(int)-c.pair_FP.astype(int)))<1e-8)
            assert(np.sum(np.abs(b.pair_MN.astype(int)-c.pair_MN.astype(int)))<1e-8)
        except AssertionError:
            print('The pairs are not deterministic')
            # Check FP pairs
            if b.pair_FP is not None and c.pair_FP is not None:
                for i in range(min(5000, len(b.pair_FP))):
                    if np.sum(np.abs(b.pair_FP[i] - c.pair_FP[i])) > 1e-8:
                        print("FP mismatch at index", i)
                        print(b.pair_FP[i])
                        print(c.pair_FP[i])
                        break
            # Check MN pairs
            if b.pair_MN is not None and c.pair_MN is not None:
                for i in range(min(5000, len(b.pair_MN))):
                    if np.sum(np.abs(b.pair_MN[i] - c.pair_MN[i])) > 1e-8:
                        print('MN mismatch at index', i)
                        print(b.pair_MN[i])
                        print(c.pair_MN[i])
                        break

def test_pacmap_metric_initialization():
    """Test that PaCMAP can be initialized with different distance metrics on supported backends."""
    for backend in get_available_backends():
        print(f"Testing metric initialization for backend: {backend}")
        
        # All backends support euclidean
        pacmap.PaCMAP(distance='euclidean', knn_backend=backend)
        
        # All backends support angular/cosine
        pacmap.PaCMAP(distance='angular', knn_backend=backend)
        
        if backend in ['annoy', 'faiss']:
            pacmap.PaCMAP(distance='manhattan', knn_backend=backend)
            
        if backend == 'annoy':
            pacmap.PaCMAP(distance='hamming', knn_backend=backend)

def test_pacmap_metric_unknown_fails():
    """Test that PaCMAP fails to initialize with unknown distance metrics."""
    with pytest.raises(NotImplementedError):
        pacmap.PaCMAP(distance='unknown')

def test_pacmap_metric_deterministic_manhattan():
    """Test PaCMAP deterministic behavior with Manhattan distance."""
    # Initialize sample data
    sample_data = np.random.normal(size=(1000, 10)) # Reduced size for speed
    
    for backend in get_available_backends():
        # Voyager does not support Manhattan
        if backend == 'voyager':
            continue
            
        print(f"Testing deterministic Manhattan for {backend}")
        b = pacmap.PaCMAP(random_state=10, distance='manhattan', knn_backend=backend)
        b_out = b.fit_transform(sample_data)
        c = pacmap.PaCMAP(random_state=10, distance='manhattan', knn_backend=backend)
        c_out = c.fit_transform(sample_data)
        
        try:
            assert np.allclose(b_out, c_out, atol=1e-8)
        except AssertionError:
            debug_nondeterminism(b, c)
            raise AssertionError(f"Backend {backend} failed deterministic Manhattan test")

def test_pacmap_metric_angular():
    """Test PaCMAP works with angular distance metric."""
    sample_data = np.random.normal(size=(1000, 10))
    
    for backend in get_available_backends():
        print(f"Testing Angular for {backend}")
        d = pacmap.PaCMAP(distance='angular', knn_backend=backend)
        d_out = d.fit_transform(sample_data)
        assert d_out.shape == (1000, 2)

def test_pacmap_metric_hamming():
    """Test PaCMAP works with hamming distance metric."""
    sample_data = np.random.normal(size=(1000, 10))
    
    for backend in get_available_backends():
        # Only Annoy supports Hamming in current implementation
        if backend != 'annoy':
            continue
            
        print(f"Testing Hamming for {backend}")
        e = pacmap.PaCMAP(distance='hamming', apply_pca=False, knn_backend=backend)
        e_out = e.fit_transform(sample_data, init='random')
        assert e_out.shape == (1000, 2)

def test_pacmap_fashion_mnist_manhattan(openml_datasets=None):
    """Test PaCMAP with Fashion-MNIST using Manhattan distance."""
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
        if backend == 'voyager':
            continue # No Manhattan support
            
        print(f"Testing FMNIST Manhattan with {backend}")
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, 
                                random_state=20, distance='manhattan', knn_backend=backend)
        embedding = reducer.fit_transform(fmnist)
        generate_figure(embedding, labels, f'test_fmnist_manhattan_{backend}')

        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, 
                                distance='manhattan', knn_backend=backend)
        embedding = reducer.fit_transform(fmnist)
        generate_figure(embedding, labels, f'test_fmnist_manhattan_noseed_{backend}')


def test_pacmap_mnist_metrics(openml_datasets=None):
    """Test PaCMAP with MNIST using different distance metrics."""
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
        print(f"Testing MNIST Metrics loop for {backend}")
        
        # Angular (All backends)
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, 
                                random_state=20, distance='angular', knn_backend=backend)
        embedding = reducer.fit_transform(mnist)
        generate_figure(embedding, labels, f'test_mnist_angular_{backend}')

        # Hamming (Annoy only)
        if backend == 'annoy':
            reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, 
                                    distance='hamming', apply_pca=True, knn_backend=backend)
            embedding = reducer.fit_transform(mnist)
            generate_figure(embedding, labels, f'test_mnist_hamming_{backend}')

        # Manhattan (Annoy and Faiss)
        if backend != 'voyager':
            reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, 
                                    random_state=20, distance='manhattan', knn_backend=backend)
            embedding = reducer.fit_transform(mnist)
            generate_figure(embedding, labels, f'test_mnist_manhattan_{backend}')

    print('Figures have been generated successfully.')


if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists("./test/output"):
        os.makedirs("./test/output")
        
    # Backward compatibility - can still run as script
    test_pacmap_metric_initialization()
    test_pacmap_metric_deterministic_manhattan()
    test_pacmap_metric_angular()
    test_pacmap_metric_hamming()
    
    # These require internet/data
    test_pacmap_fashion_mnist_manhattan()
    test_pacmap_mnist_metrics()