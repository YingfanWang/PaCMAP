'''
Tests for the `distance="precomputed"` option, which lets users pass a
user-defined square distance matrix to PaCMAP / LocalMAP instead of a feature
matrix (resolves GitHub issue #12).
'''
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances, silhouette_score

from pacmap import pacmap


@pytest.fixture(scope="module")
def blob_data():
    """Clustered data plus its labels; small enough for a dense distance matrix."""
    X, y = make_blobs(n_samples=300, centers=4, n_features=10, random_state=5)
    return X.astype(np.float32), y


@pytest.fixture(scope="module")
def euclidean_matrix(blob_data):
    """A precomputed Euclidean distance matrix for the blob data."""
    X, _ = blob_data
    return pairwise_distances(X, metric="euclidean").astype(np.float32)


def test_precomputed_initialization():
    """distance='precomputed' is accepted and stored."""
    pm = pacmap.PaCMAP(distance="precomputed")
    assert pm.distance == "precomputed"


def test_precomputed_fit_transform_shape(euclidean_matrix):
    """Fitting on an (n, n) distance matrix returns an (n, n_components) embedding."""
    pm = pacmap.PaCMAP(distance="precomputed", random_state=42)
    emb = pm.fit_transform(euclidean_matrix)
    assert emb.shape == (euclidean_matrix.shape[0], 2)
    assert np.isfinite(emb).all()


def test_precomputed_deterministic(euclidean_matrix):
    """A fixed random_state yields reproducible embeddings."""
    a = pacmap.PaCMAP(distance="precomputed", random_state=7).fit_transform(euclidean_matrix)
    b = pacmap.PaCMAP(distance="precomputed", random_state=7).fit_transform(euclidean_matrix)
    assert np.allclose(a, b, atol=1e-8)


def test_precomputed_nondeterministic(euclidean_matrix):
    """Without a random_state, two runs differ."""
    a = pacmap.PaCMAP(distance="precomputed").fit_transform(euclidean_matrix)
    b = pacmap.PaCMAP(distance="precomputed").fit_transform(euclidean_matrix)
    assert not np.allclose(a, b, atol=1e-8)


def test_precomputed_preserves_cluster_structure(blob_data, euclidean_matrix):
    """A precomputed Euclidean matrix preserves clusters about as well as raw features."""
    X, y = blob_data
    emb_feature = pacmap.PaCMAP(random_state=9).fit_transform(X)
    emb_precomputed = pacmap.PaCMAP(distance="precomputed", random_state=9).fit_transform(euclidean_matrix)

    s_feature = silhouette_score(emb_feature, y)
    s_precomputed = silhouette_score(emb_precomputed, y)
    # The precomputed path should recover well-separated blobs on its own.
    assert s_precomputed > 0.3
    # And it should be in the same ballpark as running on the raw features.
    assert s_precomputed > s_feature - 0.15


def test_precomputed_custom_metric(blob_data):
    """A non-Euclidean user-defined distance matrix (the motivating use case) works."""
    X, y = blob_data
    D = pairwise_distances(X, metric="cosine").astype(np.float32)
    emb = pacmap.PaCMAP(distance="precomputed", random_state=11).fit_transform(D)
    assert emb.shape == (X.shape[0], 2)
    assert np.isfinite(emb).all()


def test_precomputed_nonsquare_raises(blob_data):
    """A non-square matrix is rejected with a clear error."""
    X, _ = blob_data
    D = pairwise_distances(X, metric="euclidean").astype(np.float32)
    with pytest.raises(ValueError):
        pacmap.PaCMAP(distance="precomputed").fit_transform(D[:, :10])


def test_precomputed_init_pca_raises(euclidean_matrix):
    """init='pca' is unavailable for precomputed distances."""
    with pytest.raises(ValueError):
        pacmap.PaCMAP(distance="precomputed").fit_transform(euclidean_matrix, init="pca")


def test_precomputed_transform_not_implemented(euclidean_matrix):
    """transform() is not supported after fitting on a precomputed matrix."""
    pm = pacmap.PaCMAP(distance="precomputed", random_state=1)
    pm.fit(euclidean_matrix)
    with pytest.raises(NotImplementedError):
        pm.transform(euclidean_matrix)


def test_precomputed_ndarray_init(euclidean_matrix):
    """A user-supplied ndarray init is honored for precomputed distances."""
    n = euclidean_matrix.shape[0]
    rng = np.random.RandomState(0)
    init = rng.normal(size=(n, 2)).astype(np.float32)
    emb = pacmap.PaCMAP(distance="precomputed", random_state=1).fit_transform(
        euclidean_matrix, init=init)
    assert emb.shape == (n, 2)
    assert np.isfinite(emb).all()


def test_localmap_precomputed(euclidean_matrix):
    """LocalMAP also supports precomputed distance matrices."""
    lm = pacmap.LocalMAP(distance="precomputed", random_state=3)
    emb = lm.fit_transform(euclidean_matrix)
    assert emb.shape == (euclidean_matrix.shape[0], 2)
    assert np.isfinite(emb).all()


if __name__ == "__main__":
    pytest.main([__file__])
