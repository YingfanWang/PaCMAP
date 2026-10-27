"""Cosine-search rounding must not turn duplicate observations into NaN distances."""

from types import SimpleNamespace

import numpy as np
import pytest

from pacmap import pacmap


@pytest.fixture
def rounding_voyager(monkeypatch):
    """Model the small negative cosine distances observed from native Voyager."""
    class RoundingIndex:
        def __init__(self, space, num_dimensions, random_seed=1):
            self.num_elements = 0

        def add_items(self, data, num_threads=None):
            self.data = data
            self.num_elements = len(data)

        def query(self, data, k):
            # Test inputs are unit vectors. Only exact zero distances are perturbed.
            distances = (1 - data @ self.data.T).astype(np.float32)
            distances[distances == 0] = -np.finfo(np.float32).eps
            indices = np.argsort(distances, axis=1)[:, :k]
            return indices, np.take_along_axis(distances, indices, axis=1)

    monkeypatch.setattr(pacmap, "voyager", SimpleNamespace(
        Index=RoundingIndex, Space=SimpleNamespace(Cosine="cosine", Euclidean="euclidean")))


def test_training_neighbor_distances_remain_finite(rounding_voyager):
    """A duplicate's slightly negative cosine distance represents zero, not NaN."""
    data = np.repeat(np.eye(4, dtype=np.float32), 4, axis=0)
    _, _, distances = pacmap.compute_nearest_neighbors(data, 2, "angular", "voyager", 7)
    np.testing.assert_array_equal(distances, np.zeros((len(data), 2), dtype=np.float32))


def test_projection_search_receives_finite_distances(rounding_voyager, monkeypatch):
    """Query-to-basis sorting must retain identical rows as the closest matches."""
    data = np.repeat(np.eye(4, dtype=np.float32), 4, axis=0)
    captured = []
    sample = pacmap.sample_neighbors_pair_basis

    def record_distances(n, query, distances, neighbors, count):
        captured.append(distances.copy())
        return sample(n, query, distances, neighbors, count)

    monkeypatch.setattr(pacmap, "sample_neighbors_pair_basis", record_distances)
    pairs = pacmap.generate_extra_pair_basis(
        data, data[:1], 2, None, distance="angular", knn_backend="voyager")
    assert len(captured) == 1
    assert np.isfinite(captured[0]).all()
    np.testing.assert_array_equal(captured[0][:, :4], np.zeros((1, 4), dtype=np.float32))
    assert np.all(pairs[:, 1] < 4)
