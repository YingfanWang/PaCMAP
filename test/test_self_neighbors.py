"""Nearest-neighbor graphs must exclude row identity, including distance ties."""

import numpy as np
import pytest

from pacmap import pacmap


BACKENDS = [
    pytest.param("faiss", marks=pytest.mark.skipif(pacmap.faiss is None, reason="faiss unavailable")),
    pytest.param("annoy", marks=pytest.mark.skipif(pacmap.AnnoyIndex is None, reason="annoy unavailable")),
    pytest.param("voyager", marks=pytest.mark.skipif(pacmap.voyager is None, reason="voyager unavailable")),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("metric", ["euclidean", "angular"])
def test_duplicate_rows_do_not_create_self_neighbors(backend, metric):
    """Equal-distance rows must not make the query itself survive exclusion."""
    data = np.repeat(np.eye(4, dtype=np.float32), 4, axis=0)
    _, indices, distances = pacmap.compute_nearest_neighbors(data, 2, metric, backend, 7)
    assert indices.shape == distances.shape == (len(data), 2)
    assert not np.any(indices == np.arange(len(data))[:, None])
    # Each row has three other identical rows, enough to fill both neighbor slots.
    np.testing.assert_array_equal(data[indices], np.repeat(data[:, None], 2, axis=1))
    np.testing.assert_allclose(distances, 0, atol=1e-6)


def test_fit_neighbor_pairs_exclude_self_for_duplicate_observations():
    """The public embedding workflow must not waste attractive edges on self pairs."""
    if pacmap.faiss is None:
        pytest.skip("faiss unavailable")
    data = np.repeat(np.eye(4, dtype=np.float32), 4, axis=0)
    reducer = pacmap.PaCMAP(
        n_neighbors=2, random_state=7, num_iters=(2, 2, 2),
        knn_backend="faiss", numba_threads=2,
    ).fit(data, init="random")
    assert not np.any(reducer.pair_neighbors[:, 0] == reducer.pair_neighbors[:, 1])
    assert np.isfinite(reducer.embedding_).all()

