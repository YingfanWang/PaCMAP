'''
A test script ensuring PaCMAP produces deterministic results across backends.
'''
import pytest
from pacmap import pacmap
import numpy as np
from test_utils import get_available_backends, sample_data

@pytest.mark.parametrize("backend", get_available_backends())
def test_pacmap_randomness_deterministic(backend, sample_data):
    """Test that PaCMAP produces deterministic results with same random_state."""
    print(f"\nTesting determinism for backend: {backend}")
    
    # Instance 1
    p1 = pacmap.PaCMAP(
        n_components=2, 
        n_neighbors=10, 
        lr=1, 
        random_state=20, 
        apply_pca=True,
        knn_backend=backend
    )
    out1 = p1.fit_transform(sample_data, init="pca")
    
    # Instance 2
    p2 = pacmap.PaCMAP(
        n_components=2, 
        n_neighbors=10, 
        lr=1, 
        random_state=20, 
        apply_pca=True,
        knn_backend=backend
    )
    out2 = p2.fit_transform(sample_data, init="pca")

    try:
        diff = np.sum(np.abs(out1 - out2))
        assert diff < 1e-8
        print(f"Success: Backend {backend} is deterministic. Total diff: {diff:.2e}")
        
    except AssertionError:
        print(f"Failure: Backend {backend} is NOT deterministic.")
        debug_internal_pairs(p1, p2)
        raise AssertionError(f"Determinism failed for {backend}")

def debug_internal_pairs(instance1, instance2):
    """Helper to pinpoint where randomness enters the pipeline."""
    for pair_type in ['pair_FP', 'pair_MN', 'pair_neighbors']:
        p1 = getattr(instance1, pair_type, None)
        p2 = getattr(instance2, pair_type, None)
        
        if p1 is not None and p2 is not None:
            try:
                assert np.allclose(p1.astype(float), p2.astype(float), atol=1e-8)
            except AssertionError:
                print(f"  > Mismatch detected in {pair_type}!")
                # Find the first index where they differ
                mismatch_idx = np.where(~np.isclose(p1.astype(float), p2.astype(float), atol=1e-8))[0][0]
                print(f"  > First mismatch at index {mismatch_idx}:")
                print(f"    Inst1: {p1[mismatch_idx]}")
                print(f"    Inst2: {p2[mismatch_idx]}")
                break

if __name__ == "__main__":
    # Allows running via 'python test_script.py'
    pytest.main([__file__])