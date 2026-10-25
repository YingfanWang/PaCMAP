"""Snapshot selection must not change the learned trajectory or query shape."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from pacmap import pacmap


def make_reducer(estimator, **kwargs):
    return estimator(n_neighbors=3, num_iters=(1, 1, 3), random_state=7,
                     save_tree=True, **kwargs)


@pytest.mark.parametrize("estimator", [pacmap.PaCMAP, pacmap.LocalMAP])
@pytest.mark.parametrize("snapshots", [[0, 2], [1, 3]])
@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_snapshot_selection(estimator, snapshots, operation):
    """Requested snapshots equal the corresponding full history, including query-only shapes."""
    X = load_iris().data.astype(np.float32)[::3]
    if operation == "fit":
        plain = make_reducer(estimator).fit_transform(X)
        full = make_reducer(estimator, intermediate=True,
                            intermediate_snapshots=list(range(6)))
        history = full.fit_transform(X)
        partial = make_reducer(estimator, intermediate=True,
                               intermediate_snapshots=snapshots)
        selected = partial.fit_transform(X)
        np.testing.assert_array_equal(full.embedding_, plain)
        np.testing.assert_array_equal(partial.embedding_, plain)
    else:
        model = make_reducer(estimator).fit(X)
        plain = model.transform(X[:4])
        model.set_params(intermediate=True, intermediate_snapshots=list(range(6)))
        history = model.transform(X[:4])
        model.set_params(intermediate_snapshots=snapshots)
        selected = model.transform(X[:4])
        assert history.shape == (6, 4, 2)
        np.testing.assert_array_equal(history[-1], plain)
    np.testing.assert_array_equal(selected, history[snapshots])
    assert np.isfinite(selected).all()


@pytest.mark.parametrize("estimator", [pacmap.PaCMAP, pacmap.LocalMAP])
def test_default_snapshots_with_longer_optimization(estimator):
    """Optimization continues after the final default snapshot at step 450."""
    X = load_iris().data.astype(np.float32)[::3]
    kwargs = dict(n_neighbors=3, num_iters=(100, 100, 300), random_state=7)
    plain = estimator(**kwargs).fit_transform(X)
    model = estimator(intermediate=True, **kwargs)
    history = model.fit_transform(X)
    assert history.shape == (13, len(X), 2)
    np.testing.assert_array_equal(model.embedding_, plain)


@pytest.mark.parametrize("snapshots", [[], [[0, 1]], [0, 1.5], [-1, 2],
                                       [0, 3, 2], [0, 2, 2], [0, 6]])
def test_invalid_snapshots(snapshots):
    """Invalid requests fail explicitly rather than returning unwritten output slots."""
    X = load_iris().data.astype(np.float32)[::3]
    model = make_reducer(pacmap.PaCMAP, intermediate=True,
                         intermediate_snapshots=snapshots)
    with pytest.raises(ValueError, match="intermediate_snapshots"):
        model.fit_transform(X)


def test_snapshots_ignored_when_disabled():
    """Unused snapshot settings retain the previous no-history behavior."""
    X = load_iris().data.astype(np.float32)[::3]
    result = make_reducer(pacmap.PaCMAP, intermediate_snapshots=[]).fit_transform(X)
    assert result.shape == (len(X), 2)
    assert np.isfinite(result).all()


def test_transform_more_than_thirteen_snapshots():
    """Transform fills all requested slots instead of capping the cursor at twelve."""
    X = load_iris().data.astype(np.float32)[::3]
    model = pacmap.PaCMAP(n_neighbors=3, num_iters=(1, 1, 12),
                         random_state=7, save_tree=True).fit(X)
    plain = model.transform(X[:4])
    model.set_params(intermediate=True, intermediate_snapshots=list(range(15)))
    history = model.transform(X[:4])
    assert history.shape == (15, 4, 2)
    np.testing.assert_array_equal(history[-1], plain)
