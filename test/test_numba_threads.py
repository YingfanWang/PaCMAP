"""Per-estimator Numba worker limits must not leak into the caller."""
import numba
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from pacmap import pacmap


@pytest.mark.parametrize("estimator", [pacmap.PaCMAP, pacmap.LocalMAP])
@pytest.mark.parametrize(
    "threads", [1] + ([2] if numba.config.NUMBA_NUM_THREADS >= 2 else [])
)
def test_fit_transform_numba_threads(estimator, threads, monkeypatch):
    """Actual fitting and projection use the requested mask, then restore it."""
    X = load_iris().data.astype(np.float32)[::3]
    previous = numba.get_num_threads()
    observed = []
    # These are shared by PaCMAP and LocalMAP and run real compiled kernels.
    original = pacmap.update_embedding_adam
    def checked_update(*args, **kwargs):
        observed.append(numba.get_num_threads())
        return original(*args, **kwargs)
    monkeypatch.setattr(pacmap, "update_embedding_adam", checked_update)
    reducer = estimator(n_neighbors=3, num_iters=(1, 1, 2), random_state=7,
                        save_tree=True, numba_threads=threads)
    embedding = reducer.fit_transform(X)
    assert observed and set(observed) == {threads}
    assert embedding.shape == (len(X), 2)
    assert np.isfinite(embedding).all()
    assert numba.get_num_threads() == previous
    observed.clear()
    projected = reducer.transform(X[:4])
    assert observed and set(observed) == {threads}
    assert projected.shape == (4, 2)
    assert np.isfinite(projected).all()
    assert numba.get_num_threads() == previous
    assert clone(reducer).numba_threads == threads


@pytest.mark.parametrize("estimator", [pacmap.PaCMAP, pacmap.LocalMAP])
def test_numba_threads_restored_on_error(estimator, monkeypatch):
    """A failed fit restores the caller mask and preserves the original error."""
    previous = numba.get_num_threads()
    reducer = estimator(numba_threads=1)
    def fail(*args, **kwargs):
        assert numba.get_num_threads() == 1
        raise RuntimeError("failed pair sampling")
    monkeypatch.setattr(reducer, "sample_pairs", fail)
    with pytest.raises(RuntimeError, match="failed pair sampling"):
        reducer.fit(load_iris().data)
    assert numba.get_num_threads() == previous


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "2"])
def test_invalid_numba_threads(threads):
    """Invalid masks fail before input preprocessing or numerical execution."""
    previous = numba.get_num_threads()
    reducer = pacmap.PaCMAP(numba_threads=threads)
    with pytest.raises(ValueError, match="numba_threads"):
        reducer.fit(np.zeros((20, 4)))
    assert numba.get_num_threads() == previous


def test_numba_threads_above_pool_size():
    """Requests beyond the configured Numba pool fail without changing the mask."""
    previous = numba.get_num_threads()
    reducer = pacmap.PaCMAP(numba_threads=numba.config.NUMBA_NUM_THREADS + 1)
    with pytest.raises(ValueError):
        reducer.fit(np.zeros((20, 4)))
    assert numba.get_num_threads() == previous


def test_default_numba_threads(monkeypatch):
    """The default leaves the caller's current Numba mask untouched."""
    previous = numba.get_num_threads()
    reducer = pacmap.PaCMAP()
    def stop(*args, **kwargs):
        assert numba.get_num_threads() == previous
        raise RuntimeError("stop")
    monkeypatch.setattr(reducer, "sample_pairs", stop)
    with pytest.raises(RuntimeError, match="stop"):
        reducer.fit(load_iris().data)
    assert numba.get_num_threads() == previous
