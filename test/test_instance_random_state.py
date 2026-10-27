"""Estimator seeds must not depend on which other estimator was constructed last."""

import numpy as np
import pytest

from pacmap import pacmap


def _model(cls, seed, distance="euclidean"):
    return cls(
        n_neighbors=3,
        random_state=seed,
        num_iters=(2, 2, 2),
        knn_backend="faiss",
        numba_threads=2,
        save_tree=True,
        distance=distance,
    )


def _data(distance="euclidean"):
    data = np.random.default_rng(123).normal(size=(40, 6)).astype(np.float32)
    if distance == "precomputed":
        data = np.linalg.norm(data[:, None] - data[None, :], axis=-1)
    return data


@pytest.mark.parametrize("cls", [pacmap.PaCMAP, pacmap.LocalMAP])
@pytest.mark.parametrize("other_seed", [13, None])
@pytest.mark.parametrize("distance", ["euclidean", "precomputed"])
def test_other_constructor_does_not_change_fit_seed(cls, other_seed, distance):
    """Constructing an unused estimator must not change seeded fitting or pairs."""
    data = _data(distance)
    expected_model = _model(cls, 7, distance)
    expected = expected_model.fit_transform(data, init="random")
    model = _model(cls, 7, distance)
    _model(cls, other_seed, distance)
    actual = model.fit_transform(data, init="random")
    np.testing.assert_array_equal(model.pair_MN, expected_model.pair_MN)
    np.testing.assert_array_equal(model.pair_FP, expected_model.pair_FP)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("cls", [pacmap.PaCMAP, pacmap.LocalMAP])
def test_set_params_random_state_controls_fit(cls):
    """Changing the sklearn-style parameter must select the new sampling seed."""
    data = _data()
    expected = _model(cls, 7).fit_transform(data, init="random")
    model = _model(cls, 3).set_params(random_state=7)
    np.testing.assert_allclose(
        model.fit_transform(data, init="random"), expected, rtol=0, atol=1e-6
    )


def test_other_constructor_does_not_change_projection_seed():
    """A fitted model's random projection initialization uses its own seed."""
    data = _data()
    model = _model(pacmap.PaCMAP, 7).fit(data, init="random")
    expected = model.transform(data[:3], init="random")
    _model(pacmap.PaCMAP, 13)
    np.testing.assert_allclose(
        model.transform(data[:3], init="random"), expected, rtol=0, atol=1e-6
    )


def test_seeded_fit_does_not_reset_numpy_global_rng():
    """Local initialization must not reseed a caller's NumPy random stream."""
    before = np.random.get_state()
    try:
        _model(pacmap.PaCMAP, 7).fit(_data(), init="random")
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(before)


@pytest.mark.parametrize("distance", ["euclidean", "precomputed"])
def test_supplied_neighbor_pairs_use_the_estimator_seed(distance):
    """Externally supplied nearest neighbors do not bypass the estimator's seed."""
    data = _data(distance)
    reference = _model(pacmap.PaCMAP, 7, distance).fit(data, init="random")
    model = _model(pacmap.PaCMAP, 7, distance).set_params(
        pair_neighbors=reference.pair_neighbors.copy()
    )
    _model(pacmap.PaCMAP, 13)
    model.fit(data, init="random")
    np.testing.assert_array_equal(model.pair_MN, reference.pair_MN)
    np.testing.assert_array_equal(model.pair_FP, reference.pair_FP)
    np.testing.assert_allclose(model.embedding_, reference.embedding_, rtol=0, atol=1e-6)
