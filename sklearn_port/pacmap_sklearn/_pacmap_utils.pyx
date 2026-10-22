# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# cython: language_level=3
"""Cython kernels for PaCMAP.

Replaces the numba kernels of the reference implementation with code that
follows scikit-learn's compiled-extension conventions (Cython + OpenMP,
no runtime JIT).

Design constraints relative to the reference implementation:

* **Determinism independent of thread count.** Random pair sampling uses a
  counter-based generator seeded per row, so the sampled pairs depend only
  on ``seed`` and never on scheduling. The gradient is accumulated per node
  over a fixed incidence list (gather, not scatter), so there are no atomics
  and no order-dependent floating-point reductions.
* **No global state.** The seed is an explicit argument.
"""

import numpy as np

from cython.parallel cimport prange, parallel
from libc.math cimport sqrt, fabs
from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, free

ctypedef Py_ssize_t intp_t
ctypedef double float64_t

# Metric codes, mirrored in _pacmap.py
cdef enum:
    EUCLIDEAN = 0
    MANHATTAN = 1
    ANGULAR = 2
    HAMMING = 3
    PRECOMPUTED = 4

# Pair-type codes in the incidence list, mirrored in _pacmap.py
cdef enum:
    NN_PAIR = 0
    MN_PAIR = 1
    FP_PAIR = 2

# Salts so the MN and FP streams are independent for the same row
cdef uint64_t MN_SALT = 0x6A09E667F3BCC909ULL
cdef uint64_t FP_SALT = 0xBB67AE8584CAA73BULL

# Number of candidates drawn per mid-near pair (Wang et al., 2021)
cdef enum:
    N_MN_CANDIDATES = 6


# --------------------------------------------------------------------------
# Counter-based RNG (splitmix64). One uint64 of state per stream.
# --------------------------------------------------------------------------

cdef inline uint64_t _splitmix64(uint64_t* state) noexcept nogil:
    cdef uint64_t z
    state[0] += 0x9E3779B97F4A7C15ULL
    z = state[0]
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL
    return z ^ (z >> 31)


cdef inline uint64_t _seed_stream(uint64_t seed, uint64_t salt,
                                  uint64_t row) noexcept nogil:
    cdef uint64_t s = seed ^ salt ^ (row * 0xD1B54A32D192ED03ULL)
    _splitmix64(&s)
    return s


cdef inline intp_t _randint(uint64_t* state, intp_t n) noexcept nogil:
    # Modulo bias is < n / 2**64, negligible for any feasible n.
    return <intp_t>(_splitmix64(state) % <uint64_t>n)


# --------------------------------------------------------------------------
# Distances. Only used to rank mid-near candidates, so any monotone
# transform of the true distance is acceptable.
# --------------------------------------------------------------------------

cdef inline float64_t _dist(const float64_t[:, ::1] X, intp_t i, intp_t j,
                            int metric) noexcept nogil:
    cdef intp_t k, p = X.shape[1]
    cdef float64_t acc = 0.0, nx = 0.0, ny = 0.0, diff
    if metric == PRECOMPUTED:
        return X[i, j]
    if metric == EUCLIDEAN:
        for k in range(p):
            diff = X[i, k] - X[j, k]
            acc += diff * diff
        return acc  # squared distance, monotone in the distance
    if metric == MANHATTAN:
        for k in range(p):
            acc += fabs(X[i, k] - X[j, k])
        return acc
    if metric == ANGULAR:
        for k in range(p):
            acc += X[i, k] * X[j, k]
            nx += X[i, k] * X[i, k]
            ny += X[j, k] * X[j, k]
        nx = sqrt(nx) if nx > 1e-40 else 1e-20
        ny = sqrt(ny) if ny > 1e-40 else 1e-20
        return -acc / (nx * ny)  # monotone in sqrt(2 - 2 cos)
    # HAMMING
    for k in range(p):
        if X[i, k] != X[j, k]:
            acc += 1.0
    return acc


cdef inline bint _contains(const intp_t* arr, intp_t length,
                           intp_t value) noexcept nogil:
    cdef intp_t k
    for k in range(length):
        if arr[k] == value:
            return True
    return False


# --------------------------------------------------------------------------
# Pair sampling
# --------------------------------------------------------------------------

def sample_mn_pairs(const float64_t[:, ::1] X, intp_t n_samples, intp_t n_mn,
                    uint64_t seed, int metric, int n_threads):
    """Sample mid-near pairs.

    For each point ``i`` and each of its ``n_mn`` slots, draw up to 6 distinct
    candidates uniformly (excluding ``i`` and the mid-near partners already
    chosen for ``i``) and keep the second closest.

    ``X`` is the preprocessed data, or the dense distance matrix when
    ``metric == PRECOMPUTED``. Requires ``n_mn <= n_samples - 2``.
    """
    out = np.empty((n_samples * n_mn, 2), dtype=np.intp)
    cdef intp_t[:, ::1] pairs = out
    cdef intp_t i, j, t, c, n_cand, cand, best, second
    cdef float64_t d, d_best, d_second
    cdef uint64_t state
    cdef intp_t* cands      # per-thread candidate buffer
    cdef intp_t* chosen     # per-thread record of row i's previous picks

    if n_mn == 0:
        return out

    with nogil, parallel(num_threads=n_threads):
        cands = <intp_t*> malloc(N_MN_CANDIDATES * sizeof(intp_t))
        chosen = <intp_t*> malloc(n_mn * sizeof(intp_t))
        for i in prange(n_samples, schedule="static"):
            state = _seed_stream(seed, MN_SALT, <uint64_t>i)
            for j in range(n_mn):
                n_cand = n_samples - 1 - j
                if n_cand > N_MN_CANDIDATES:
                    n_cand = N_MN_CANDIDATES
                c = 0
                while c < n_cand:
                    cand = _randint(&state, n_samples)
                    if cand == i:
                        continue
                    if _contains(cands, c, cand):
                        continue
                    if _contains(chosen, j, cand):
                        continue
                    cands[c] = cand
                    c = c + 1
                # Second-closest candidate (closest if only one is available).
                best = cands[0]
                d_best = _dist(X, i, best, metric)
                second = -1
                d_second = 0.0
                for t in range(1, n_cand):
                    cand = cands[t]
                    d = _dist(X, i, cand, metric)
                    if d < d_best:
                        second = best
                        d_second = d_best
                        best = cand
                        d_best = d
                    elif second == -1 or d < d_second:
                        second = cand
                        d_second = d
                if second == -1:
                    second = best
                chosen[j] = second
                pairs[i * n_mn + j, 0] = i
                pairs[i * n_mn + j, 1] = second
        free(cands)
        free(chosen)
    return out


def sample_fp_pairs(const intp_t[:, ::1] neighbors, intp_t n_fp,
                    uint64_t seed, int n_threads):
    """Sample further pairs: ``n_fp`` distinct uniform draws per point,
    excluding the point itself and its selected nearest neighbors.

    Requires ``n_fp <= n_samples - 1 - neighbors.shape[1]``.
    """
    cdef intp_t n_samples = neighbors.shape[0]
    cdef intp_t n_nn = neighbors.shape[1]
    out = np.empty((n_samples * n_fp, 2), dtype=np.intp)
    cdef intp_t[:, ::1] pairs = out
    cdef intp_t i, k, t, cand
    cdef uint64_t state

    if n_fp == 0:
        return out

    for i in prange(n_samples, nogil=True, num_threads=n_threads,
                    schedule="static"):
        state = _seed_stream(seed, FP_SALT, <uint64_t>i)
        k = 0
        while k < n_fp:
            cand = _randint(&state, n_samples)
            if cand == i:
                continue
            if _contains(&neighbors[i, 0], n_nn, cand):
                continue
            t = 0
            while t < k:
                if pairs[i * n_fp + t, 1] == cand:
                    break
                t = t + 1
            if t < k:
                continue
            pairs[i * n_fp + k, 0] = i
            pairs[i * n_fp + k, 1] = cand
            k = k + 1
    return out


# --------------------------------------------------------------------------
# Gradient
# --------------------------------------------------------------------------

def pacmap_gradient(const float64_t[:, ::1] Y,
                    const intp_t[::1] indptr,
                    const intp_t[::1] other,
                    const signed char[::1] pair_type,
                    float64_t w_nn, float64_t w_mn, float64_t w_fp,
                    float64_t[:, ::1] grad,
                    float64_t[::1] node_loss,
                    int n_threads):
    """Gradient of the PaCMAP loss, written into ``grad``.

    The incidence list (``indptr``, ``other``, ``pair_type``) holds every pair
    once per endpoint, so ``grad[a]`` is a gather over a fixed sequence and
    the result is bitwise identical for any ``n_threads``.

    With :math:`\\tilde d_{ab} = 1 + \\|y_a - y_b\\|^2`, each pair contributes
    ``coef * (y_a - y_b)`` to ``grad[a]``:

    =====  ==================================  =====================================
    pair   loss                                coef
    =====  ==================================  =====================================
    NN     :math:`w\\tilde d/(10+\\tilde d)`      :math:`20w/(10+\\tilde d)^2`
    MN     :math:`w\\tilde d/(10^4+\\tilde d)`    :math:`2\\cdot10^4 w/(10^4+\\tilde d)^2`
    FP     :math:`w/(1+\\tilde d)`               :math:`-2w/(1+\\tilde d)^2`
    =====  ==================================  =====================================

    ``node_loss[a]`` receives the loss summed over pairs incident to ``a``;
    the total loss is ``node_loss.sum() / 2``.
    """
    cdef intp_t n = Y.shape[0], dim = Y.shape[1]
    cdef intp_t a, b, p, k
    cdef float64_t d, diff, coef, loss
    cdef signed char t
    cdef bint skip_mn = (w_mn == 0.0)

    for a in prange(n, nogil=True, num_threads=n_threads, schedule="static"):
        for k in range(dim):
            grad[a, k] = 0.0
        loss = 0.0
        for p in range(indptr[a], indptr[a + 1]):
            t = pair_type[p]
            if skip_mn and t == MN_PAIR:
                continue
            b = other[p]
            d = 1.0
            for k in range(dim):
                diff = Y[a, k] - Y[b, k]
                d = d + diff * diff
            if t == NN_PAIR:
                loss = loss + w_nn * d / (10.0 + d)
                coef = w_nn * 20.0 / ((10.0 + d) * (10.0 + d))
            elif t == MN_PAIR:
                loss = loss + w_mn * d / (10000.0 + d)
                coef = w_mn * 20000.0 / ((10000.0 + d) * (10000.0 + d))
            else:
                loss = loss + w_fp / (1.0 + d)
                coef = -w_fp * 2.0 / ((1.0 + d) * (1.0 + d))
            for k in range(dim):
                grad[a, k] = grad[a, k] + coef * (Y[a, k] - Y[b, k])
        node_loss[a] = loss
