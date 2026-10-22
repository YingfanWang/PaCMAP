"""Pairwise Controlled Manifold Approximation Projection (PaCMAP).

Laid out as ``sklearn/manifold/_pacmap.py`` would be: only numpy, scipy and
scikit-learn at runtime, compiled kernels in ``_pacmap_utils.pyx``.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real

import numpy as np

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_non_negative, validate_data

from ._pacmap_utils import pacmap_gradient, sample_fp_pairs, sample_mn_pairs

# Metric codes, mirrored in _pacmap_utils.pyx
_METRIC_CODES = {
    "euclidean": 0,
    "manhattan": 1,
    "angular": 2,
    "hamming": 3,
    "precomputed": 4,
}
# Pair-type codes, mirrored in _pacmap_utils.pyx
_NN, _MN, _FP = 0, 1, 2

# Constants of the reference implementation (Wang et al., 2021)
_PCA_DIM = 100  # target dimension of the preprocessing projection
_N_EXTRA_NEIGHBORS = 50  # extra candidates searched before scaled re-ranking
_W_MN_INIT = 1000.0  # mid-near weight at iteration 0
_ADAM_BETA1, _ADAM_BETA2, _ADAM_EPS = 0.9, 0.999, 1e-7


def _phase_weights(itr, n_iter_phase1, n_iter_phase2):
    """(w_nn, w_mn, w_fp) at iteration ``itr`` of the three-phase schedule."""
    if itr < n_iter_phase1:
        frac = itr / n_iter_phase1
        return 2.0, (1.0 - frac) * _W_MN_INIT + frac * 3.0, 1.0
    if itr < n_iter_phase1 + n_iter_phase2:
        return 3.0, 3.0, 1.0
    return 1.0, 0.0, 1.0


def _incidence_list(pairs, pair_types, n_samples):
    """CSR incidence structure listing each pair once per endpoint.

    Ordering is a stable sort on the endpoint index, so it is a pure function
    of the input and gradient accumulation order is fixed.
    """
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    other = np.concatenate([pairs[:, 1], pairs[:, 0]])
    types = np.concatenate([pair_types, pair_types])
    order = np.argsort(rows, kind="stable")
    indptr = np.zeros(n_samples + 1, dtype=np.intp)
    np.cumsum(np.bincount(rows, minlength=n_samples), out=indptr[1:])
    return (
        indptr,
        np.ascontiguousarray(other[order], dtype=np.intp),
        np.ascontiguousarray(types[order], dtype=np.int8),
    )


class PaCMAP(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Pairwise Controlled Manifold Approximation Projection.

    PaCMAP embeds data by optimizing a loss over three sets of point pairs:
    nearest-neighbor (NN) pairs that preserve local structure, mid-near (MN)
    pairs that preserve global structure, and further (FP) pairs that keep
    unrelated points apart. Optimization runs in three phases with a fixed
    weight schedule: MN pairs dominate early to set the global layout, then
    are switched off so NN and FP pairs refine local structure.

    Read more in the :ref:`User Guide <pacmap>`.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    n_neighbors : int or None, default=10
        Number of NN pairs per point. If None, uses 10 for ``n_samples <=
        10000`` and ``round(10 + 15 * (log10(n_samples) - 4))`` otherwise.

    mn_ratio : float, default=0.5
        Number of MN pairs per point, as a ratio of ``n_neighbors``.

    fp_ratio : float, default=2.0
        Number of FP pairs per point, as a ratio of ``n_neighbors``.

    metric : {"euclidean", "manhattan", "angular", "hamming", \
            "precomputed"}, default="euclidean"
        Metric in the input space. "angular" is
        :math:`\\sqrt{2 - 2\\cos\\theta}`. If "precomputed", ``X`` is a dense
        square matrix of pairwise distances.

    learning_rate : float, default=1.0
        Adam step size.

    num_iters : int or tuple of 3 int, default=(100, 100, 250)
        Iterations in each of the three phases. An int ``k`` means
        ``(100, 100, k)``.

    init : {"pca", "random"} or ndarray of shape (n_samples, n_components), \
            default="pca"
        Initialization of the embedding. "pca" cannot be used with
        ``metric="precomputed"``. An array is standardized and scaled by
        1e-4.

    apply_pca : bool, default=True
        If True and ``n_features > 100``, center the data and project it onto
        its top 100 principal components before searching for neighbors.
        Ignored for ``metric`` in {"hamming", "precomputed"}.

    random_state : int, RandomState instance or None, default=None
        Determines pair sampling, PCA and random initialization. Pass an int
        for reproducible output across calls. Output does not depend on the
        number of threads.

    n_jobs : int, default=None
        Number of parallel jobs for the neighbors search. The compiled
        kernels use OpenMP and are controlled by ``OMP_NUM_THREADS`` or
        :mod:`threadpoolctl`.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        The embedding.

    loss_ : float
        Loss at the start of the last iteration.

    n_iter_ : int
        Total number of iterations run.

    n_neighbors_, n_mn_, n_fp_ : int
        Number of NN, MN and FP pairs per point actually used, after
        adjustment for small ``n_samples``.

    pair_neighbors_, pair_mn_, pair_fp_ : ndarray of shape (n_pairs, 2)
        Sampled pairs, as rows ``(i, j)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    TSNE : T-distributed Stochastic Neighbor Embedding.
    Isomap : Isomap embedding.

    References
    ----------
    .. [1] Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021).
       Understanding How Dimension Reduction Tools Work: An Empirical
       Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data
       Visualization. Journal of Machine Learning Research, 22(201), 1-73.

    Examples
    --------
    >>> import numpy as np
    >>> from pacmap_sklearn import PaCMAP
    >>> X = np.random.RandomState(0).rand(100, 5)
    >>> X_embedded = PaCMAP(random_state=0).fit_transform(X)
    >>> X_embedded.shape
    (100, 2)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],
        "mn_ratio": [Interval(Real, 0, None, closed="left")],
        "fp_ratio": [Interval(Real, 0, None, closed="neither")],
        "metric": [StrOptions(set(_METRIC_CODES))],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "num_iters": [Interval(Integral, 0, None, closed="left"), tuple],
        "init": [StrOptions({"pca", "random"}), np.ndarray],
        "apply_pca": ["boolean"],
        "random_state": ["random_state"],
        "n_jobs": [None, Integral],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_components=2,
        *,
        n_neighbors=10,
        mn_ratio=0.5,
        fp_ratio=2.0,
        metric="euclidean",
        learning_rate=1.0,
        num_iters=(100, 100, 250),
        init="pca",
        apply_pca=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.mn_ratio = mn_ratio
        self.fp_ratio = fp_ratio
        self.metric = metric
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.init = init
        self.apply_pca = apply_pca
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #

    def _check_num_iters(self):
        if isinstance(self.num_iters, tuple):
            if len(self.num_iters) != 3 or not all(
                isinstance(k, Integral) and k >= 0 for k in self.num_iters
            ):
                raise ValueError(
                    "num_iters must be a non-negative int or a tuple of 3 "
                    f"non-negative ints, got {self.num_iters!r}."
                )
            return tuple(int(k) for k in self.num_iters)
        return (100, 100, int(self.num_iters))

    def _n_pairs(self, n_samples):
        """Pairs per point, adjusted as in the reference implementation."""
        if self.n_neighbors is None:
            if n_samples <= 10000:
                n_nn = 10
            else:
                n_nn = int(round(10 + 15 * (np.log10(n_samples) - 4)))
        else:
            n_nn = self.n_neighbors
        n_mn = int(round(n_nn * self.mn_ratio))
        n_fp = int(round(n_nn * self.fp_ratio))

        n_nn = min(n_nn, n_samples - 1)
        n_fp = min(n_fp, n_samples - 1 - n_nn)
        n_mn = min(n_mn, n_samples - 1)
        if n_nn + n_mn + n_fp >= n_samples:
            # Too few points: shrink all three in the requested proportions.
            scale = n_samples / (1.0 + self.mn_ratio + self.fp_ratio)
            n_nn = max(int(scale), 1)
            n_mn = int(scale * self.mn_ratio)
            n_fp = max(int(scale * self.fp_ratio), 1)
        # Hard feasibility limits of the samplers.
        n_mn = max(min(n_mn, n_samples - 2), 0)
        n_fp = min(n_fp, n_samples - 1 - n_nn)

        if n_fp < 1:
            raise ValueError(
                f"PaCMAP needs at least one further pair per point; got "
                f"n_samples={n_samples}, n_neighbors={n_nn}, "
                f"fp_ratio={self.fp_ratio}. Increase fp_ratio or the number "
                "of samples."
            )
        return n_nn, n_mn, n_fp

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def _preprocess(self, X, random_state):
        """Return (data for neighbor search, PCA init or None)."""
        want_pca_init = isinstance(self.init, str) and self.init == "pca"
        n_samples, n_features = X.shape
        if (
            self.metric != "hamming"
            and n_features > _PCA_DIM
            and self.apply_pca
        ):
            X = X - X.mean(axis=0)
            X = TruncatedSVD(
                n_components=_PCA_DIM, random_state=random_state
            ).fit_transform(X)
            pca_init = X[:, : self.n_components] if want_pca_init else None
        else:
            X = X.copy()
            X -= X.min()
            x_max = X.max()
            if x_max > 0:
                X /= x_max
            X -= X.mean(axis=0)
            pca_init = None
            if want_pca_init:
                k = min(self.n_components, n_samples, n_features)
                pca_init = PCA(
                    n_components=k, random_state=random_state
                ).fit_transform(X)
        if pca_init is not None and pca_init.shape[1] < self.n_components:
            # Fewer informative directions than output dimensions: pad with
            # small noise so the extra coordinates are not stuck at zero.
            pad = 1e-2 * random_state.standard_normal(
                (n_samples, self.n_components - pca_init.shape[1])
            )
            pca_init = np.hstack([pca_init, pad])
        return np.ascontiguousarray(X, dtype=np.float64), pca_init

    def _nearest_neighbor_pairs(self, X, n_nn):
        """NN pairs: exact kNN, then re-rank by locally scaled distance."""
        n_samples = X.shape[0]
        n_search = min(n_nn + _N_EXTRA_NEIGHBORS, n_samples - 1)
        sk_metric = {"angular": "cosine"}.get(self.metric, self.metric)
        knn = NearestNeighbors(
            n_neighbors=n_search, metric=sk_metric, n_jobs=self.n_jobs
        ).fit(X)
        dist, idx = knn.kneighbors(return_distance=True)
        if self.metric == "angular":
            dist = np.sqrt(2.0 * np.maximum(dist, 0.0))

        # sigma_i: mean distance to the 4th-6th neighbors.
        lo, hi = min(3, n_search - 1), min(6, n_search)
        sigma = np.maximum(dist[:, lo:hi].mean(axis=1), 1e-10)
        scaled = dist**2 / sigma[:, None] / sigma[idx]
        order = np.argsort(scaled, axis=1, kind="stable")[:, :n_nn]
        neighbors = np.ascontiguousarray(
            np.take_along_axis(idx, order, axis=1), dtype=np.intp
        )
        pairs = np.column_stack(
            [np.repeat(np.arange(n_samples, dtype=np.intp), n_nn), neighbors.ravel()]
        )
        return pairs, neighbors

    def _initial_embedding(self, n_samples, pca_init, random_state):
        if isinstance(self.init, np.ndarray):
            init = np.asarray(self.init, dtype=np.float64)
            if init.shape != (n_samples, self.n_components):
                raise ValueError(
                    "init must have shape (n_samples, n_components) = "
                    f"{(n_samples, self.n_components)}, got {init.shape}."
                )
            return 1e-4 * StandardScaler().fit_transform(init)
        if self.init == "random":
            return 1e-4 * random_state.standard_normal(
                (n_samples, self.n_components)
            )
        return 1e-2 * pca_init

    def _optimize(self, Y, pairs, pair_types, num_iters, n_threads):
        n_samples = Y.shape[0]
        indptr, other, types = _incidence_list(pairs, pair_types, n_samples)
        grad = np.empty_like(Y)
        node_loss = np.empty(n_samples)
        m = np.zeros_like(Y)
        v = np.zeros_like(Y)
        step = np.empty_like(Y)
        n_iter = sum(num_iters)
        loss = np.nan
        for itr in range(n_iter):
            w_nn, w_mn, w_fp = _phase_weights(itr, num_iters[0], num_iters[1])
            pacmap_gradient(
                Y, indptr, other, types, w_nn, w_mn, w_fp, grad, node_loss,
                n_threads,
            )
            loss = 0.5 * node_loss.sum()
            # Adam with the bias correction folded into the step size.
            lr_t = (
                self.learning_rate
                * np.sqrt(1.0 - _ADAM_BETA2 ** (itr + 1))
                / (1.0 - _ADAM_BETA1 ** (itr + 1))
            )
            m += (1.0 - _ADAM_BETA1) * (grad - m)
            v += (1.0 - _ADAM_BETA2) * (grad * grad - v)
            np.sqrt(v, out=step)
            step += _ADAM_EPS
            np.divide(m, step, out=step)
            step *= lr_t
            Y -= step
            if self.verbose and (itr == 0 or (itr + 1) % 10 == 0):
                print(f"[PaCMAP] Iteration {itr + 1}: loss = {loss:.6f}")
        return Y, loss, n_iter

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Compute the embedding of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data, or pairwise distances if ``metric="precomputed"``.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """Compute the embedding of `X` and return it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data, or pairwise distances if ``metric="precomputed"``.

        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data.
        """
        precomputed = self.metric == "precomputed"
        X = validate_data(
            self, X, dtype=np.float64, ensure_min_samples=3, copy=False
        )
        n_samples = X.shape[0]
        if precomputed:
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    'With metric="precomputed", X must be a square distance '
                    f"matrix; got shape {X.shape}."
                )
            check_non_negative(X, "PaCMAP.fit(). With metric='precomputed', X")
            if isinstance(self.init, str) and self.init == "pca":
                raise ValueError(
                    'The parameter init="pca" cannot be used with '
                    'metric="precomputed".'
                )
        num_iters = self._check_num_iters()
        random_state = check_random_state(self.random_state)
        n_threads = _openmp_effective_n_threads()

        n_nn, n_mn, n_fp = self._n_pairs(n_samples)
        if precomputed:
            X_search, pca_init = np.ascontiguousarray(X), None
        else:
            X_search, pca_init = self._preprocess(X, random_state)

        # One seed for both samplers; they use independent salted streams.
        seed = int(random_state.randint(np.iinfo(np.int32).max))
        pair_nn, neighbors = self._nearest_neighbor_pairs(X_search, n_nn)
        pair_mn = sample_mn_pairs(
            X_search, n_samples, n_mn, seed, _METRIC_CODES[self.metric], n_threads
        )
        pair_fp = sample_fp_pairs(neighbors, n_fp, seed, n_threads)

        Y = np.ascontiguousarray(
            self._initial_embedding(n_samples, pca_init, random_state),
            dtype=np.float64,
        )
        pairs = np.concatenate([pair_nn, pair_mn, pair_fp])
        pair_types = np.concatenate(
            [
                np.full(len(pair_nn), _NN, dtype=np.int8),
                np.full(len(pair_mn), _MN, dtype=np.int8),
                np.full(len(pair_fp), _FP, dtype=np.int8),
            ]
        )
        Y, loss, n_iter = self._optimize(Y, pairs, pair_types, num_iters, n_threads)

        self.embedding_ = Y
        self.loss_ = float(loss)
        self.n_iter_ = n_iter
        self.n_neighbors_, self.n_mn_, self.n_fp_ = n_nn, n_mn, n_fp
        self.pair_neighbors_ = pair_nn
        self.pair_mn_ = pair_mn
        self.pair_fp_ = pair_fp
        return self.embedding_

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.embedding_.shape[1]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.metric == "precomputed"
        tags.input_tags.positive_only = self.metric == "precomputed"
        return tags
