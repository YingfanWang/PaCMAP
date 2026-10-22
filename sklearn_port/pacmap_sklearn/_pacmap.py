"""PaCMAP and LocalMAP.

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

from ._pacmap_utils import (
    build_incidence,
    pacmap_gradient,
    resample_fp_nearby,
    sample_fp_pairs,
    sample_mn_pairs,
)

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

# LocalMAP (Wang et al., 2025): FP pairs are resampled every this many
# iterations during phase 3.
_LOCALMAP_RESAMPLE_EVERY = 10

# Rashomon neighbor refinement (Sun et al., 2026): size m of the candidate
# neighbor set that the k refined neighbors are chosen from.
_N_RASHOMON_CANDIDATES = 50


def _phase_weights(itr, n_iter_phase1, n_iter_phase2):
    """(w_nn, w_mn, w_fp) at iteration ``itr`` of the three-phase schedule."""
    if itr < n_iter_phase1:
        frac = itr / n_iter_phase1
        return 2.0, (1.0 - frac) * _W_MN_INIT + frac * 3.0, 1.0
    if itr < n_iter_phase1 + n_iter_phase2:
        return 3.0, 3.0, 1.0
    return 1.0, 0.0, 1.0


def _incidence_list(pairs, pair_types, n_samples):
    """CSR incidence structure listing each pair once per endpoint."""
    return build_incidence(
        np.ascontiguousarray(pairs, dtype=np.intp),
        np.ascontiguousarray(pair_types, dtype=np.int8),
        n_samples,
    )


def _pair_types(n_nn, n_mn, n_fp):
    return np.concatenate(
        [
            np.full(n_nn, _NN, dtype=np.int8),
            np.full(n_mn, _MN, dtype=np.int8),
            np.full(n_fp, _FP, dtype=np.int8),
        ]
    )


class _Adam:
    """Adam with the bias correction folded into the step size, as in the
    reference implementation."""

    def __init__(self, shape, learning_rate):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.step = np.empty(shape)
        self.learning_rate = learning_rate

    def update(self, Y, grad, itr):
        lr_t = (
            self.learning_rate
            * np.sqrt(1.0 - _ADAM_BETA2 ** (itr + 1))
            / (1.0 - _ADAM_BETA1 ** (itr + 1))
        )
        self.m += (1.0 - _ADAM_BETA1) * (grad - self.m)
        self.v += (1.0 - _ADAM_BETA2) * (grad * grad - self.v)
        np.sqrt(self.v, out=self.step)
        self.step += _ADAM_EPS
        np.divide(self.m, self.step, out=self.step)
        self.step *= lr_t
        Y -= self.step


_BASE_PARAMETER_CONSTRAINTS = {
    "n_components": [Interval(Integral, 1, None, closed="left")],
    "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],
    "mn_ratio": [Interval(Real, 0, None, closed="left")],
    "fp_ratio": [Interval(Real, 0, None, closed="neither")],
    "metric": [StrOptions(set(_METRIC_CODES))],
    "learning_rate": [Interval(Real, 0, None, closed="neither")],
    "num_iters": [Interval(Integral, 0, None, closed="left"), tuple],
    "init": [StrOptions({"pca", "random"}), np.ndarray],
    "apply_pca": ["boolean"],
    "n_rashomon": [Interval(Integral, 2, None, closed="left"), None],
    "rashomon_penalty": [Interval(Real, 0, None, closed="left")],
    "random_state": ["random_state"],
    "n_jobs": [None, Integral],
    "verbose": ["verbose"],
}


class _BasePaCMAP(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Shared machinery of PaCMAP and LocalMAP. Subclasses define
    ``__init__`` and ``_optimize``."""

    _verbose_name = "PaCMAP"

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
                f"{type(self).__name__} needs at least one further pair per "
                f"point; got n_samples={n_samples}, n_neighbors={n_nn}, "
                f"fp_ratio={self.fp_ratio}. Increase fp_ratio or the number "
                "of samples."
            )
        return n_nn, n_mn, n_fp

    # ------------------------------------------------------------------ #
    # Fit stages
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

    def _knn(self, X, n_nn):
        """Exact kNN (self excluded), ``n_nn + 50`` neighbors per point."""
        n_samples = X.shape[0]
        n_search = min(n_nn + _N_EXTRA_NEIGHBORS, n_samples - 1)
        sk_metric = {"angular": "cosine"}.get(self.metric, self.metric)
        knn = NearestNeighbors(
            n_neighbors=n_search, metric=sk_metric, n_jobs=self.n_jobs
        ).fit(X)
        dist, idx = knn.kneighbors(return_distance=True)
        if self.metric == "angular":
            dist = np.sqrt(2.0 * np.maximum(dist, 0.0))
        return dist, idx

    @staticmethod
    def _scaled_neighbors(dist, idx, n_nn):
        """Re-rank kNN candidates by locally scaled distance (reference)."""
        n_search = dist.shape[1]
        # sigma_i: mean distance to the 4th-6th neighbors.
        lo, hi = min(3, n_search - 1), min(6, n_search)
        sigma = np.maximum(dist[:, lo:hi].mean(axis=1), 1e-10)
        scaled = dist**2 / sigma[:, None] / sigma[idx]
        order = np.argsort(scaled, axis=1, kind="stable")[:, :n_nn]
        return np.ascontiguousarray(
            np.take_along_axis(idx, order, axis=1), dtype=np.intp
        )

    @staticmethod
    def _neighbor_pairs(neighbors):
        n_samples, n_nn = neighbors.shape
        return np.column_stack(
            [
                np.repeat(np.arange(n_samples, dtype=np.intp), n_nn),
                neighbors.ravel(),
            ]
        )

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

    def _run(self, X_search, pca_init, neighbors, n_mn, n_fp, num_iters,
             random_state, n_threads):
        """Sample MN / FP pairs, initialize and optimize one embedding."""
        n_samples = X_search.shape[0]
        # One seed for all samplers; they use independent salted streams.
        seed = int(random_state.randint(np.iinfo(np.int32).max))
        pair_nn = self._neighbor_pairs(neighbors)
        pair_mn = sample_mn_pairs(
            X_search, n_samples, n_mn, seed, _METRIC_CODES[self.metric], n_threads
        )
        pair_fp = sample_fp_pairs(neighbors, n_fp, seed, n_threads)
        Y = np.ascontiguousarray(
            self._initial_embedding(n_samples, pca_init, random_state),
            dtype=np.float64,
        )
        Y, loss, n_iter, pair_fp = self._optimize(
            Y, pair_nn, pair_mn, pair_fp, neighbors, num_iters, seed, n_threads
        )
        return Y, loss, n_iter, pair_nn, pair_mn, pair_fp

    def _rashomon_neighbors(self, X_search, pca_init, dist, idx, n_nn, n_mn,
                            n_fp, num_iters, random_state, n_threads):
        """Trustworthy-neighbor refinement (Sun et al., 2026, Algorithm 2).

        Candidates are the ``m = 50`` nearest neighbors in the input space.
        ``n_rashomon`` embeddings fitted with different seeds serve both as
        calibration set and as Rashomon set. For candidate pair ``(i, j)``
        with embedding distances :math:`d^{(t)}_{ij}`:

        .. math::

            \\mu_{ij} = \\tfrac1T\\sum_t d^{(t)}_{ij},\\quad
            \\sigma_{ij}^2 = \\tfrac1T\\sum_t (d^{(t)}_{ij}-\\mu_{ij})^2,\\quad
            m_{ij} = \\max_t d^{(t)}_{ij},\\quad
            M = \\operatorname{mean}_{i,j}\\mu_{ij}

        .. math::

            \\bar\\Psi_{ij} = \\tfrac1T\\sum_t \\Psi^{(t)}_{ij}
            = \\begin{cases} \\mu_{ij} + \\lambda\\sigma_{ij} & \\mu_{ij} < M \\\\
                             m_{ij} + \\lambda\\sigma_{ij} & \\text{otherwise}
              \\end{cases}

        and the ``n_nn`` candidates with the lowest :math:`\\bar\\Psi_{ij}`
        become the NN partners of ``i``.
        """
        n_samples = X_search.shape[0]
        n_cand = min(max(_N_RASHOMON_CANDIDATES, n_nn), idx.shape[1])
        candidates = np.ascontiguousarray(idx[:, :n_cand], dtype=np.intp)
        default_neighbors = self._scaled_neighbors(dist, idx, n_nn)

        s1 = np.zeros((n_samples, n_cand))
        s2 = np.zeros((n_samples, n_cand))
        d_max = np.full((n_samples, n_cand), -np.inf)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_rashomon)
        for t, seed in enumerate(seeds):
            Y_t, *_ = self._run(
                X_search, pca_init, default_neighbors, n_mn, n_fp, num_iters,
                np.random.RandomState(seed), n_threads,
            )
            d = np.linalg.norm(Y_t[:, None, :] - Y_t[candidates], axis=2)
            s1 += d
            s2 += d * d
            np.maximum(d_max, d, out=d_max)
            if self.verbose:
                print(f"[{self._verbose_name}] Rashomon run {t + 1}/{self.n_rashomon}")

        T = float(self.n_rashomon)
        mu = s1 / T
        sigma = np.sqrt(np.maximum(s2 / T - mu * mu, 0.0))
        threshold = mu.mean()
        psi = np.where(mu < threshold, mu, d_max) + self.rashomon_penalty * sigma

        order = np.argsort(psi, axis=1, kind="stable")[:, :n_nn]
        neighbors = np.ascontiguousarray(
            np.take_along_axis(candidates, order, axis=1), dtype=np.intp
        )
        return neighbors, candidates, psi

    def _fit(self, X):
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
            check_non_negative(
                X, f"{type(self).__name__}.fit(). With metric='precomputed', X"
            )
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

        dist, idx = self._knn(X_search, n_nn)
        if self.n_rashomon is None:
            neighbors = self._scaled_neighbors(dist, idx, n_nn)
        else:
            neighbors, candidates, psi = self._rashomon_neighbors(
                X_search, pca_init, dist, idx, n_nn, n_mn, n_fp, num_iters,
                random_state, n_threads,
            )
            self.rashomon_candidates_ = candidates
            self.rashomon_scores_ = psi

        Y, loss, n_iter, pair_nn, pair_mn, pair_fp = self._run(
            X_search, pca_init, neighbors, n_mn, n_fp, num_iters,
            random_state, n_threads,
        )
        self.embedding_ = Y
        self.loss_ = float(loss)
        self.n_iter_ = n_iter
        self.n_neighbors_, self.n_mn_, self.n_fp_ = n_nn, n_mn, n_fp
        self.pair_neighbors_ = pair_nn
        self.pair_mn_ = pair_mn
        self.pair_fp_ = pair_fp
        return self.embedding_

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
        self._fit(X)
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
        return self._fit(X)

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.embedding_.shape[1]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.metric == "precomputed"
        tags.input_tags.positive_only = self.metric == "precomputed"
        return tags


_SHARED_PARAMS_DOC = """
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
"""

_RASHOMON_PARAMS_DOC = """
    n_rashomon : int >= 2 or None, default=None
        If set, refine the NN pairs with the trustworthy-neighbor procedure
        of [Sun2026]_ (its Algorithm 2) before the final fit: fit
        ``n_rashomon`` embeddings with different seeds, score each of the 50
        nearest input-space candidates of every point by its mean
        upper-confidence distance :math:`\\bar\\Psi_{ij}`, and keep the
        ``n_neighbors`` lowest-scoring candidates as NN partners. Costs
        ``n_rashomon`` extra optimizations; the neighbor search is shared.

    rashomon_penalty : float, default=1.0
        Penalty :math:`\\lambda` on the across-seed standard deviation in
        :math:`\\Psi`. Only used if ``n_rashomon`` is set.

    random_state : int, RandomState instance or None, default=None
        Determines pair sampling, PCA, random initialization and the seeds
        of the Rashomon runs. Pass an int for reproducible output across
        calls. Output does not depend on the number of threads.

    n_jobs : int, default=None
        Number of parallel jobs for the neighbors search. The compiled
        kernels use OpenMP and are controlled by ``OMP_NUM_THREADS`` or
        :mod:`threadpoolctl`.

    verbose : int, default=0
        Verbosity level.
"""

_ATTRIBUTES_DOC = """
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
        Pairs used in the final optimization, as rows ``(i, j)``.

    rashomon_candidates_ : ndarray of shape (n_samples, n_candidates)
        Candidate NN partners scored by the Rashomon refinement. Only
        defined if ``n_rashomon`` is set.

    rashomon_scores_ : ndarray of shape (n_samples, n_candidates)
        :math:`\\bar\\Psi_{ij}` of each candidate; lower is more trustworthy.
        Only defined if ``n_rashomon`` is set.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
"""

_REFERENCES_DOC = """
    .. [Wang2021] Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021).
       Understanding How Dimension Reduction Tools Work: An Empirical
       Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data
       Visualization. Journal of Machine Learning Research, 22(201), 1-73.

    .. [Sun2026] Sun, Y., Huang, H., Parikh, G. R., & Rudin, C. (2026).
       The Rashomon Effect for Visualizing High-Dimensional Data. AISTATS.
"""


class PaCMAP(_BasePaCMAP):
    __doc__ = (
        """Pairwise Controlled Manifold Approximation Projection.

    PaCMAP embeds data by optimizing a loss over three sets of point pairs:
    nearest-neighbor (NN) pairs that preserve local structure, mid-near (MN)
    pairs that preserve global structure, and further (FP) pairs that keep
    unrelated points apart. Optimization runs in three phases with a fixed
    weight schedule: MN pairs dominate early to set the global layout, then
    are switched off so NN and FP pairs refine local structure.

    Read more in the :ref:`User Guide <pacmap>`.

    Parameters
    ----------"""
        + _SHARED_PARAMS_DOC
        + _RASHOMON_PARAMS_DOC
        + _ATTRIBUTES_DOC
        + """
    See Also
    --------
    LocalMAP : PaCMAP with local graph adjustment in the last phase.
    TSNE : T-distributed Stochastic Neighbor Embedding.

    References
    ----------"""
        + _REFERENCES_DOC
        + """
    Examples
    --------
    >>> import numpy as np
    >>> from pacmap_sklearn import PaCMAP
    >>> X = np.random.RandomState(0).rand(100, 5)
    >>> X_embedded = PaCMAP(random_state=0).fit_transform(X)
    >>> X_embedded.shape
    (100, 2)
    """
    )

    _parameter_constraints: dict = dict(_BASE_PARAMETER_CONSTRAINTS)

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
        n_rashomon=None,
        rashomon_penalty=1.0,
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
        self.n_rashomon = n_rashomon
        self.rashomon_penalty = rashomon_penalty
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _optimize(self, Y, pair_nn, pair_mn, pair_fp, neighbors, num_iters,
                  seed, n_threads):
        n_samples = Y.shape[0]
        pairs = np.concatenate([pair_nn, pair_mn, pair_fp])
        types = _pair_types(len(pair_nn), len(pair_mn), len(pair_fp))
        indptr, other, types = _incidence_list(pairs, types, n_samples)
        grad = np.empty_like(Y)
        node_loss = np.empty(n_samples)
        adam = _Adam(Y.shape, self.learning_rate)
        n_iter = sum(num_iters)
        loss = np.nan
        for itr in range(n_iter):
            w_nn, w_mn, w_fp = _phase_weights(itr, num_iters[0], num_iters[1])
            pacmap_gradient(
                Y, indptr, other, types, w_nn, w_mn, w_fp, grad, node_loss,
                n_threads,
            )
            loss = 0.5 * node_loss.sum()
            adam.update(Y, grad, itr)
            if self.verbose and (itr == 0 or (itr + 1) % 10 == 0):
                print(f"[PaCMAP] Iteration {itr + 1}: loss = {loss:.6f}")
        return Y, loss, n_iter, pair_fp


class LocalMAP(_BasePaCMAP):
    __doc__ = (
        """LocalMAP: PaCMAP with local graph adjustment.

    Runs PaCMAP's first two phases unchanged. In the third phase it
    (a) multiplies the NN gradient coefficient by
    :math:`(\\text{low\\_dist\\_thres}/2)/\\sqrt{\\tilde d_{ij}}`, pulling
    close neighbors harder, and (b) every 10 iterations resamples the FP
    pairs among points within ``low_dist_thres`` of each point in the
    current embedding, so repulsion acts between nearby clusters.

    Phase-3 behavior starts at iteration ``n1 + n2 + 1`` and resampling at
    the first multiple of 10 after ``n1 + n2``, matching the reference
    implementation.

    Parameters
    ----------"""
        + _SHARED_PARAMS_DOC
        + """
    low_dist_thres : float, default=10.0
        Embedding-space radius within which FP pairs are resampled in
        phase 3; also sets the NN gradient multiplier.
"""
        + _RASHOMON_PARAMS_DOC
        + _ATTRIBUTES_DOC
        + """
    See Also
    --------
    PaCMAP : Pairwise Controlled Manifold Approximation Projection.

    References
    ----------"""
        + _REFERENCES_DOC
        + """
    .. [Wang2025] Wang, Y., Sun, Y., Huang, H., & Rudin, C. (2025).
       Dimension Reduction with Locally Adjusted Graphs. AAAI, 39(20).

    Examples
    --------
    >>> import numpy as np
    >>> from pacmap_sklearn import LocalMAP
    >>> X = np.random.RandomState(0).rand(100, 5)
    >>> X_embedded = LocalMAP(random_state=0).fit_transform(X)
    >>> X_embedded.shape
    (100, 2)
    """
    )

    _verbose_name = "LocalMAP"
    _parameter_constraints: dict = {
        **_BASE_PARAMETER_CONSTRAINTS,
        "low_dist_thres": [Interval(Real, 0, None, closed="neither")],
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
        low_dist_thres=10.0,
        n_rashomon=None,
        rashomon_penalty=1.0,
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
        self.low_dist_thres = low_dist_thres
        self.n_rashomon = n_rashomon
        self.rashomon_penalty = rashomon_penalty
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _optimize(self, Y, pair_nn, pair_mn, pair_fp, neighbors, num_iters,
                  seed, n_threads):
        n_samples = Y.shape[0]
        types = _pair_types(len(pair_nn), len(pair_mn), len(pair_fp))
        indptr, other, inc_types = _incidence_list(
            np.concatenate([pair_nn, pair_mn, pair_fp]), types, n_samples
        )
        grad = np.empty_like(Y)
        node_loss = np.empty(n_samples)
        adam = _Adam(Y.shape, self.learning_rate)
        n_iter = sum(num_iters)
        phase3_after = num_iters[0] + num_iters[1]
        nn_recip_coef = self.low_dist_thres / 2.0
        loss = np.nan
        for itr in range(n_iter):
            w_nn, w_mn, w_fp = _phase_weights(itr, num_iters[0], num_iters[1])
            # Strict inequality as in the reference implementation.
            in_phase3 = itr > phase3_after
            pacmap_gradient(
                Y, indptr, other, inc_types, w_nn, w_mn, w_fp, grad, node_loss,
                n_threads, nn_recip_coef if in_phase3 else 0.0,
            )
            loss = 0.5 * node_loss.sum()
            adam.update(Y, grad, itr)
            if in_phase3 and itr % _LOCALMAP_RESAMPLE_EVERY == 0:
                pair_fp = resample_fp_nearby(
                    Y, neighbors, pair_fp, self.low_dist_thres, seed, itr,
                    n_threads,
                )
                indptr, other, inc_types = _incidence_list(
                    np.concatenate([pair_nn, pair_mn, pair_fp]), types, n_samples
                )
            if self.verbose and (itr == 0 or (itr + 1) % 10 == 0):
                print(f"[LocalMAP] Iteration {itr + 1}: loss = {loss:.6f}")
        return Y, loss, n_iter, pair_fp
