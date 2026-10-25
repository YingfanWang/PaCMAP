"""Round-trip Voyager-backed estimators through both public load interfaces."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from pacmap import pacmap

pytestmark = pytest.mark.skipif(pacmap.voyager is None, reason="Voyager is optional")


@pytest.mark.parametrize("estimator", [pacmap.PaCMAP, pacmap.LocalMAP])
@pytest.mark.parametrize("explicit_paths", [False, True])
def test_voyager_roundtrip(estimator, explicit_paths, tmp_path):
    """Loaded index queries and transforms still work after the input file closes."""
    X = load_iris().data.astype(np.float32)[::3]
    reducer = estimator(n_neighbors=3, num_iters=(1, 1, 2), random_state=7,
                        save_tree=True, knn_backend="voyager")
    reducer.fit(X)
    ids, distances = reducer.tree.query(X[:4], k=3, num_threads=1)
    prefix = tmp_path / "reducer"
    pacmap.save(reducer, str(prefix))
    if explicit_paths:
        restored = pacmap.load(reducer_path=str(prefix.with_suffix(".pkl")),
                               index_path=str(prefix.with_suffix(".voyager")))
    else:
        restored = pacmap.load(str(prefix))
    restored_ids, restored_distances = restored.tree.query(X[:4], k=3, num_threads=1)
    np.testing.assert_array_equal(restored_ids, ids)
    np.testing.assert_array_equal(restored_distances, distances)
    np.testing.assert_array_equal(restored.embedding_, reducer.embedding_)
    assert np.isfinite(restored.transform(X[:4])).all()
