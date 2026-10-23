"""scikit-learn common estimator checks."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from pacmap_sklearn import LocalMAP, PaCMAP

FAST = dict(num_iters=(10, 10, 20))


@parametrize_with_checks(
    [
        PaCMAP(**FAST),
        PaCMAP(metric="angular", **FAST),
        PaCMAP(metric="manhattan", n_components=3, **FAST),
        PaCMAP(metric="precomputed", init="random", **FAST),
        PaCMAP(n_rashomon=2, **FAST),
        LocalMAP(**FAST),
        LocalMAP(metric="precomputed", init="random", **FAST),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
