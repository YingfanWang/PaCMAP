"""scikit-learn common estimator checks."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from pacmap_sklearn import PaCMAP


@parametrize_with_checks(
    [
        PaCMAP(num_iters=(10, 10, 20)),
        PaCMAP(metric="angular", num_iters=(10, 10, 20)),
        PaCMAP(metric="manhattan", n_components=3, num_iters=(10, 10, 20)),
        PaCMAP(metric="precomputed", init="random", num_iters=(10, 10, 20)),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
