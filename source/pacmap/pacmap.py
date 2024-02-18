import numba
import time
import math
import datetime
import warnings
import os

import numpy as np
import pickle as pkl

from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import preprocessing
from annoy import AnnoyIndex

global _RANDOM_STATE
_RANDOM_STATE = None


@numba.njit("f4(f4[:])", cache=True)
def l2_norm(x):
    """
    L2 norm of a vector.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def euclid_dist(x1, x2):
    """
    Euclidean distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def manhattan_dist(x1, x2):
    """
    Manhattan distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += np.abs(x1[i] - x2[i])
    return result


@numba.njit("f4(f4[:],f4[:])", cache=True)
def angular_dist(x1, x2):
    """
    Angular (i.e. cosine) distance between two vectors.
    """
    x1_norm = np.maximum(l2_norm(x1), 1e-20)
    x2_norm = np.maximum(l2_norm(x2), 1e-20)
    result = 0.0
    for i in range(x1.shape[0]):
        result += x1[i] * x2[i]
    return np.sqrt(2.0 - 2.0 * result / x1_norm / x2_norm)


@numba.njit("f4(f4[:],f4[:])", cache=True)
def hamming_dist(x1, x2):
    """
    Hamming distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        if x1[i] != x2[i]:
            result += 1.0
    return result


@numba.njit(cache=True)
def calculate_dist(x1, x2, distance_index):
    if distance_index == 0:  # euclidean
        return euclid_dist(x1, x2)
    elif distance_index == 1:  # manhattan
        return manhattan_dist(x1, x2)
    elif distance_index == 2:  # angular
        return angular_dist(x1, x2)
    elif distance_index == 3:  # hamming
        return hamming_dist(x1, x2)


@numba.njit("i4[:](i4,i4,i4[:])", nogil=True, cache=True)
def sample_FP(n_samples, maximum, reject_ind):
    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(maximum)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(reject_ind.shape[0]):
                if j == reject_ind[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result


@numba.njit("i4[:,:](f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True, cache=True)
def sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors):
    n = X.shape[0]
    pair_neighbors = np.empty((n*n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i*n_neighbors + j][0] = i
            pair_neighbors[i*n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors


@numba.njit("i4[:,:](i4,f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True, cache=True)
def sample_neighbors_pair_basis(n_basis, X, scaled_dist, nbrs, n_neighbors):
    '''Sample Nearest Neighbor pairs for additional data.'''
    n = X.shape[0]
    pair_neighbors = np.empty((n*n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i*n_neighbors + j][0] = n_basis + i
            pair_neighbors[i*n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors


@numba.njit("i4[:,:](f4[:,:],i4,i4)", nogil=True, cache=True)
def sample_MN_pair(X, n_MN, option=0):
    '''Sample Mid Near pairs.'''
    n = X.shape[0]
    pair_MN = np.empty((n*n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for jj in range(n_MN):
            sampled = np.random.randint(0, n, 6)
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = calculate_dist(
                    X[i], X[sampled[t]], distance_index=option)
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i*n_MN + jj][0] = i
            pair_MN[i*n_MN + jj][1] = picked
    return pair_MN


@numba.njit("i4[:,:](f4[:,:],i4,i4,i4)", nogil=True, cache=True)
def sample_MN_pair_deterministic(X, n_MN, random_state, option=0):
    '''Sample Mid Near pairs using the given random state.'''
    n = X.shape[0]
    pair_MN = np.empty((n*n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for jj in range(n_MN):
            # Shifting the seed to prevent sampling the same pairs
            np.random.seed(random_state + i * n_MN + jj)
            sampled = np.random.randint(0, n, 6)
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = calculate_dist(
                    X[i], X[sampled[t]], distance_index=option)
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i*n_MN + jj][0] = i
            pair_MN[i*n_MN + jj][1] = picked
    return pair_MN


@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP):
    '''Sample Further pairs.'''
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            FP_index = sample_FP(
                n_FP, n, pair_neighbors[i*n_neighbors: i*n_neighbors + n_neighbors][1])
            pair_FP[i*n_FP + k][0] = i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP


@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4,i4)", parallel=True, nogil=True, cache=True)
def sample_FP_pair_deterministic(X, pair_neighbors, n_neighbors, n_FP, random_state):
    '''Sample Further pairs using the given random state.'''
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            np.random.seed(random_state+i*n_FP+k)
            FP_index = sample_FP(
                n_FP, n, pair_neighbors[i*n_neighbors: i*n_neighbors + n_neighbors][1])
            pair_FP[i*n_FP + k][0] = i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP


@numba.njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True, cache=True)
def scale_dist(knn_distance, sig, nbrs):
    '''Scale the distance'''
    n, num_neighbors = knn_distance.shape
    scaled_dist = np.zeros((n, num_neighbors), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(num_neighbors):
            scaled_dist[i, j] = knn_distance[i, j] ** 2 / \
                sig[i] / sig[nbrs[i, j]]
    return scaled_dist


@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,f4,f4,i4)", parallel=True, nogil=True, cache=True)
def update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr):
    '''Update the embedding with the gradient'''
    n, dim = Y.shape
    lr_t = lr * math.sqrt(1.0 - beta2**(itr+1)) / (1.0 - beta1**(itr+1))
    for i in numba.prange(n):
        for d in numba.prange(dim):
            m[i][d] += (1 - beta1) * (grad[i][d] - m[i][d])
            v[i][d] += (1 - beta2) * (grad[i][d]**2 - v[i][d])
            Y[i][d] -= lr_t * m[i][d]/(math.sqrt(v[i][d]) + 1e-7)


@numba.njit("f4[:,:](f4[:,:],i4[:,:],i4[:,:],i4[:,:],f4,f4,f4)", parallel=True, nogil=True, cache=True)
def pacmap_grad(Y, pair_neighbors, pair_MN, pair_FP, w_neighbors, w_MN, w_FP):
    '''Calculate the gradient for pacmap embedding given the particular set of weights.'''
    n, dim = Y.shape
    grad = np.zeros((n+1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(4, dtype=np.float32)
    # NN
    for t in range(pair_neighbors.shape[0]):
        i = pair_neighbors[t, 0]
        j = pair_neighbors[t, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[0] += w_neighbors * (d_ij/(10. + d_ij))
        w1 = w_neighbors * (20./(10. + d_ij) ** 2)
        for d in range(dim):
            grad[i, d] += w1 * y_ij[d]
            grad[j, d] -= w1 * y_ij[d]
    # MN
    for tt in range(pair_MN.shape[0]):
        i = pair_MN[tt, 0]
        j = pair_MN[tt, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i][d] - Y[j][d]
            d_ij += y_ij[d] ** 2
        loss[1] += w_MN * d_ij/(10000. + d_ij)
        w = w_MN * 20000./(10000. + d_ij) ** 2
        for d in range(dim):
            grad[i, d] += w * y_ij[d]
            grad[j, d] -= w * y_ij[d]
    # FP
    for ttt in range(pair_FP.shape[0]):
        i = pair_FP[ttt, 0]
        j = pair_FP[ttt, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[2] += w_FP * 1./(1. + d_ij)
        w1 = w_FP * 2./(1. + d_ij) ** 2
        for d in range(dim):
            grad[i, d] -= w1 * y_ij[d]
            grad[j, d] += w1 * y_ij[d]
    grad[-1, 0] = loss.sum()
    return grad


@numba.njit("f4[:,:](f4[:,:],i4[:,:],f4)", parallel=True, nogil=True, cache=True)
def pacmap_grad_fit(Y, pair_XP, w_neighbors):
    '''Calculate the gradient for pacmap embedding given the particular set of weights.'''
    n, dim = Y.shape
    grad = np.zeros((n+1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(4, dtype=np.float32)
    # For case where extra samples are added to the dataset
    for tx in range(pair_XP.shape[0]):
        i = pair_XP[tx, 0]
        j = pair_XP[tx, 1]
        d_ij = 1.0
        for d in range(dim):
            y_ij[d] = Y[i, d] - Y[j, d]
            d_ij += y_ij[d] ** 2
        loss[3] += w_neighbors * (d_ij/(10. + d_ij))
        w1 = w_neighbors * (20./(10. + d_ij) ** 2)
        # w1 = 1. * 2./(1. + d_ij) ** 2 # original formula
        for d in range(dim):
            # just compute the gradient for new point
            grad[i, d] += w1 * y_ij[d]
        #    grad[j, d] += w1 * y_ij[d]

    grad[-1, 0] = loss.sum()
    return grad


def find_weight(w_MN_init, itr, *, num_iters):
    '''Find the corresponding weight given the index of an iteration'''
    (phase_1_iters, phase_2_iters, _) = num_iters

    if itr < phase_1_iters:
        w_MN = (1 - itr/phase_1_iters) * w_MN_init + itr/phase_1_iters * 3.0
        w_neighbors = 2.0
        w_FP = 1.0
    elif itr < phase_1_iters + phase_2_iters:
        w_MN = 3.0
        w_neighbors = 3
        w_FP = 1
    else:
        w_MN = 0.0
        w_neighbors = 1.
        w_FP = 1.
    return w_MN, w_neighbors, w_FP


def preprocess_X(X, distance, apply_pca, verbose, seed, high_dim, low_dim):
    '''Preprocess a dataset.
    '''
    tsvd = None
    pca_solution = False
    if distance != "hamming" and high_dim > 100 and apply_pca:
        xmin = 0  # placeholder
        xmax = 0  # placeholder
        xmean = np.mean(X, axis=0)
        X -= np.mean(X, axis=0)
        tsvd = TruncatedSVD(n_components=100, random_state=seed)
        X = tsvd.fit_transform(X)
        pca_solution = True
        print_verbose("Applied PCA, the dimensionality becomes 100", verbose)
    else:
        xmin, xmax = (np.min(X), np.max(X))
        X -= xmin
        X /= xmax
        xmean = np.mean(X, axis=0)
        X -= xmean
        tsvd = PCA(n_components=low_dim, random_state=seed)  # for init only
        tsvd.fit(X)
        print_verbose("X is normalized", verbose)
    return X, pca_solution, tsvd, xmin, xmax, xmean


def preprocess_X_new(X, distance, xmin, xmax, xmean, tsvd, apply_pca, verbose):
    '''Preprocess a new dataset, given the information extracted from the basis.
    '''
    _, high_dim = X.shape
    if distance != "hamming" and high_dim > 100 and apply_pca:
        X -= xmean  # original xmean
        X = tsvd.transform(X)
        print_verbose(
            "Applied PCA, the dimensionality becomes 100 for new dataset.", verbose)
    else:
        X -= xmin
        X /= xmax
        X -= xmean
        print_verbose("X is normalized.", verbose)
    return X


def distance_to_option(distance='euclidean'):
    '''A helper function that translates distance metric to int options.
    Such a translation is useful for numba acceleration.
    '''
    if distance == 'euclidean':
        option = 0
    elif distance == 'manhattan':
        option = 1
    elif distance == 'angular':
        option = 2
    elif distance == 'hamming':
        option = 3
    else:
        raise NotImplementedError('Distance other than euclidean, manhattan,' +
                                  'angular or hamming is not supported')
    return option


def print_verbose(msg, verbose, **kwargs):
    if verbose:
        print(msg, **kwargs)


def generate_extra_pair_basis(basis, X,
                              n_neighbors,
                              tree: AnnoyIndex,
                              distance='euclidean',
                              verbose=True
                              ):
    '''Generate pairs that connects the extra set of data to the fitted basis.
    '''
    npr, dimp = X.shape

    assert (basis is not None or tree is not None), "If the annoyindex is not cached, the original dataset must be provided."

    # Build the tree again if not cached
    if tree is None:
        n, dim = basis.shape
        assert dimp == dim, "The dimension of the original dataset is different from the new one's."
        tree = AnnoyIndex(dim, metric=distance)
        if _RANDOM_STATE is not None:
            tree.set_seed(_RANDOM_STATE)
        for i in range(n):
            tree.add_item(i, basis[i, :])
        tree.build(20)
    else:
        n = tree.get_n_items()

    n_neighbors_extra = min(n_neighbors + 50, n - 1)
    nbrs = np.zeros((npr, n_neighbors_extra), dtype=np.int32)
    knn_distances = np.empty((npr, n_neighbors_extra), dtype=np.float32)

    for i in range(npr):
        nbrs[i, :], knn_distances[i, :] = tree.get_nns_by_vector(
            X[i, :], n_neighbors_extra, include_distances=True)

    print_verbose("Found nearest neighbor", verbose)
    # sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
    # print_verbose("Calculated sigma", verbose)

    # Debug
    # print_verbose(f"Sigma is of the scale of {sig.shape}", verbose)
    # print_verbose(f"KNN dist is of shape scale of {knn_distances.shape}", verbose)
    # print_verbose(f"nbrs max: {nbrs.max()}", verbose)

    # scaling the distances is not possible since we don't always track the basis
    # scaled_dist = scale_dist(knn_distances, sig, nbrs)
    print_verbose("Found scaled dist", verbose)

    pair_neighbors = sample_neighbors_pair_basis(
        n, X, knn_distances, nbrs, n_neighbors)
    return pair_neighbors


def generate_pair(
        X,
        n_neighbors,
        n_MN,
        n_FP,
        distance='euclidean',
        verbose=True
):
    '''Generate pairs for the dataset.
    '''
    n, dim = X.shape
    # sample more neighbors than needed
    n_neighbors_extra = min(n_neighbors + 50, n - 1)
    tree = AnnoyIndex(dim, metric=distance)
    if _RANDOM_STATE is not None:
        tree.set_seed(_RANDOM_STATE)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(20)

    option = distance_to_option(distance=distance)

    nbrs = np.zeros((n, n_neighbors_extra), dtype=np.int32)
    knn_distances = np.empty((n, n_neighbors_extra), dtype=np.float32)

    for i in range(n):
        nbrs_ = tree.get_nns_by_item(i, n_neighbors_extra + 1)
        nbrs[i, :] = nbrs_[1:]
        for j in range(n_neighbors_extra):
            knn_distances[i, j] = tree.get_distance(i, nbrs[i, j])
    print_verbose("Found nearest neighbor", verbose)
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
    print_verbose("Calculated sigma", verbose)
    scaled_dist = scale_dist(knn_distances, sig, nbrs)
    print_verbose("Found scaled dist", verbose)
    pair_neighbors = sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors)
    if _RANDOM_STATE is None:
        pair_MN = sample_MN_pair(X, n_MN, option)
        pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    else:
        pair_MN = sample_MN_pair_deterministic(X, n_MN, _RANDOM_STATE, option)
        pair_FP = sample_FP_pair_deterministic(
            X, pair_neighbors, n_neighbors, n_FP, _RANDOM_STATE)
    return pair_neighbors, pair_MN, pair_FP, tree


def generate_pair_no_neighbors(
        X,
        n_neighbors,
        n_MN,
        n_FP,
        pair_neighbors,
        distance='euclidean',
        verbose=True
):
    '''Generate mid-near pairs and further pairs for a given dataset.
    This function is useful when the nearest neighbors comes from a given set.
    '''
    option = distance_to_option(distance=distance)

    if _RANDOM_STATE is None:
        pair_MN = sample_MN_pair(X, n_MN, option)
        pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    else:
        pair_MN = sample_MN_pair_deterministic(X, n_MN, _RANDOM_STATE, option)
        pair_FP = sample_FP_pair_deterministic(
            X, pair_neighbors, n_neighbors, n_FP, _RANDOM_STATE)
    return pair_neighbors, pair_MN, pair_FP


def pacmap(
        X,
        n_dims,
        pair_neighbors,
        pair_MN,
        pair_FP,
        lr,
        num_iters,
        Yinit,
        verbose,
        intermediate,
        inter_snapshots,
        pca_solution,
        tsvd=None
):
    start_time = time.time()
    n, _ = X.shape

    if intermediate:
        intermediate_states = np.empty(
            (len(inter_snapshots), n, n_dims), dtype=np.float32)
    else:
        intermediate_states = None

    # Initialize the embedding
    if isinstance(Yinit, np.ndarray):
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = scaler.transform(Yinit) * 0.0001
    elif Yinit is None or (isinstance(Yinit, str) and Yinit == "pca"):
        if pca_solution:
            Y = 0.01 * X[:, :n_dims]
        else:
            Y = 0.01 * tsvd.transform(X).astype(np.float32)
    elif (isinstance(Yinit, str) and Yinit == "random"):  # random or hamming distance
        if _RANDOM_STATE is not None:
            np.random.seed(_RANDOM_STATE)
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    else:
        raise ValueError((f"The argument init is of the type {type(Yinit)}. "
                          "Currently, PaCMAP only supports user supplied "
                          "numpy.ndarray object as input, or one of "
                          "['pca', 'random']."))

    # Initialize parameters for optimizer
    w_MN_init = 1000.
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate and inter_snapshots[0] == 0:
        itr_ind = 1  # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose(
        (pair_neighbors.shape, pair_MN.shape, pair_FP.shape), verbose)

    num_iters_total = sum(num_iters)

    for itr in range(num_iters_total):
        w_MN, w_neighbors, w_FP = find_weight(w_MN_init, itr, num_iters=num_iters)

        grad = pacmap_grad(Y, pair_neighbors, pair_MN,
                           pair_FP, w_neighbors, w_MN, w_FP)
        C = grad[-1, 0]
        if verbose and itr == 0:
            print(f"Initial Loss: {C}")
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)

        if intermediate:
            if (itr + 1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f" % (itr + 1, C), verbose)

    elapsed = time.time() - start_time
    print_verbose(f"Elapsed time: {elapsed:.2f}s", verbose)

    return Y, intermediate_states, pair_neighbors, pair_MN, pair_FP


def pacmap_fit(
        X,
        embedding,
        n_dims,
        pair_XP,
        lr,
        num_iters,
        Yinit,
        verbose,
        intermediate,
        inter_snapshots,
        pca_solution=False,
        tsvd=None
):
    '''PaCMAP optimization for new data.
    '''
    start_time = time.time()
    n, high_dim = X.shape

    if intermediate:
        intermediate_states = np.empty(
            (len(inter_snapshots), n, n_dims), dtype=np.float32)
    else:
        intermediate_states = None

    # Initialize the embedding
    if isinstance(Yinit, np.ndarray):
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = np.concatenate([embedding, scaler.transform(Yinit) * 0.0001])
    elif Yinit is None or (isinstance(Yinit, str) and Yinit == "pca"):
        if pca_solution:
            Y = np.concatenate([embedding, 0.01 * X[:, :n_dims]])
        else:
            Y = np.concatenate(
                [embedding, 0.01 * tsvd.transform(X).astype(np.float32)])
    elif (isinstance(Yinit, str) and Yinit == "random"):  # random or hamming distance
        if _RANDOM_STATE is not None:
            np.random.seed(_RANDOM_STATE)
        Y = np.concatenate(
            [embedding, 0.0001 * np.random.normal(size=[X.shape[0], n_dims]).astype(np.float32)])
    else:
        raise ValueError((f"The argument init is of the type {type(Yinit)}. "
                          "Currently, PaCMAP only supports user supplied "
                          "numpy.ndarray object as input, or one of "
                          "['pca', 'random']."))

    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate and inter_snapshots[0] == 0:
        itr_ind = 1  # move counter to one step
        intermediate_states[0, :, :] = Y

    print_verbose(pair_XP.shape, verbose)

    num_iters_total = sum(num_iters)

    for itr in range(num_iters_total):
        _, w_neighbors, _ = find_weight(0, itr, num_iters=num_iters)

        grad = pacmap_grad_fit(Y, pair_XP, w_neighbors)
        C = grad[-1, 0]
        if verbose and itr == 0:
            print(f"Initial Loss: {C}")
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)

        if intermediate:
            if (itr+1) == inter_snapshots[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
                if itr_ind > 12:
                    itr_ind -= 1
        if (itr + 1) % 10 == 0:
            print_verbose("Iteration: %4d, Loss: %f" % (itr + 1, C), verbose)

    elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
    print_verbose("Elapsed time: %s" % (elapsed), verbose)
    return Y, intermediate_states


def save(instance, common_prefix: str):
    '''
    Save PaCMAP instance to a location specified by the user.
    PaCMAP use ANNOY for graph construction, which cannot be pickled. We provide
    this function as an alternative to save a PaCMAP instance by storing the
    ANNOY instance and other parts of PaCMAP separately.
    '''
    extra_info = ""
    if instance.save_tree:
        # Save the AnnoyIndex
        instance.tree.save(f"{common_prefix}.ann")
        temp_tree = instance.tree
        instance.tree = None # Remove the tree for pickle
        extra_info = f", and the Annoy Index is saved at {common_prefix}.ann"
    
    # Save the other parts
    with open(f"{common_prefix}.pkl", "wb") as fp:
        pkl.dump(instance, fp)

    print(f"The PaCMAP instance is successfully saved at {common_prefix}.pkl{extra_info}.")
    print(f"To load the instance again, please do `pacmap.load({common_prefix})`.")

    if instance.save_tree:
        # Reload the AnnoyIndex
        instance.tree = temp_tree # reload the annoy index
        assert instance.tree is not None


def load(common_prefix: str):
    '''
    Load PaCMAP instance from a location specified by the user.
    '''
    with open(f"{common_prefix}.pkl", "rb") as fp:
        instance = pkl.load(fp)
    if os.path.exists(f"{common_prefix}.ann"):
        instance.tree = AnnoyIndex(instance.num_dimensions, instance.distance)
        instance.tree.load(f"{common_prefix}.ann") # mmap the file

    return instance


class PaCMAP(BaseEstimator):
    '''Pairwise Controlled Manifold Approximation.

    Maps high-dimensional dataset to a low-dimensional embedding.
    This class inherits the sklearn BaseEstimator, and we tried our best to
    follow the sklearn api. For details of this method, please refer to our publication:
    https://www.jmlr.org/papers/volume22/20-1061/20-1061.pdf

    Parameters
    ---------
    n_components: int, default=2
        Dimensions of the embedded space. We recommend to use 2 or 3.

    n_neighbors: int, default=10
        Number of neighbors considered for nearest neighbor pairs for local structure preservation.

    MN_ratio: float, default=0.5
        Ratio of mid near pairs to nearest neighbor pairs (e.g. n_neighbors=10, MN_ratio=0.5 --> 5 Mid near pairs)
        Mid near pairs are used for global structure preservation.

    FP_ratio: float, default=2.0
        Ratio of further pairs to nearest neighbor pairs (e.g. n_neighbors=10, FP_ratio=2 --> 20 Further pairs)
        Further pairs are used for both local and global structure preservation.

    pair_neighbors: numpy.ndarray, optional
        Nearest neighbor pairs constructed from a previous run or from outside functions.

    pair_MN: numpy.ndarray, optional
        Mid near pairs constructed from a previous run or from outside functions.

    pair_FP: numpy.ndarray, optional
        Further pairs constructed from a previous run or from outside functions.

    distance: string, default="euclidean"
        Distance metric used for high-dimensional space. Allowed metrics include euclidean, manhattan, angular, hamming.

    lr: float, default=1.0
        Learning rate of the Adam optimizer for embedding.

    num_iters: tuple[int, int, int] or int, default=(100, 100, 250)
        Tuple with number of iterations for the optimization of embedding for each stage.
        If a single integer is provided, this parameter will be set to (100, 100, num_iters), following the original
        implementation.

    verbose: bool, default=False
        Whether to print additional information during initialization and fitting.

    apply_pca: bool, default=True
        Whether to apply PCA on the data before pair construction.

    intermediate: bool, default=False
        Whether to return intermediate state of the embedding during optimization.
        If True, returns a series of embedding during different stages of optimization.

    intermediate_snapshots: list[int], optional
        The index of step where an intermediate snapshot of the embedding is taken.
        If intermediate sets to True, the default value will be [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450]

    random_state: int, optional
        Random state for the pacmap instance.
        Setting random state is useful for repeatability.

    save_tree: bool, default=False
        Whether to save the annoy index tree after finding the nearest neighbor pairs.
        Default to False for memory saving. Setting this option to True can make `transform()` method faster.
    '''

    def __init__(self,
                 n_components=2,
                 n_neighbors=10,
                 MN_ratio=0.5,
                 FP_ratio=2.0,
                 pair_neighbors=None,
                 pair_MN=None,
                 pair_FP=None,
                 distance="euclidean",
                 lr=1.0,
                 num_iters=(100, 100, 250),
                 verbose=False,
                 apply_pca=True,
                 intermediate=False,
                 intermediate_snapshots=[
                     0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450],
                 random_state=None,
                 save_tree=False
                 ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.pair_neighbors = pair_neighbors
        self.pair_MN = pair_MN
        self.pair_FP = pair_FP
        self.distance = distance
        self.lr = lr
        self.num_iters = num_iters if hasattr(num_iters, "__len__") else (100, 100, num_iters)
        self.apply_pca = apply_pca
        self.verbose = verbose
        self.intermediate = intermediate
        self.save_tree = save_tree
        self.intermediate_snapshots = intermediate_snapshots

        global _RANDOM_STATE
        if random_state is not None:
            assert(isinstance(random_state, int))
            self.random_state = random_state
            _RANDOM_STATE = random_state  # Set random state for numba functions
            warnings.warn(f'Warning: random state is set to {_RANDOM_STATE}')
        else:
            try:
                if _RANDOM_STATE is not None:
                    warnings.warn(f'Warning: random state is removed')
            except NameError:
                pass
            self.random_state = 0
            _RANDOM_STATE = None  # Reset random state

        if self.n_components < 2:
            raise ValueError(
                "The number of projection dimensions must be at least 2.")
        if self.lr <= 0:
            raise ValueError("The learning rate must be larger than 0.")
        if self.distance == "hamming" and apply_pca:
            warnings.warn(
                "apply_pca = True for Hamming distance. This option will be ignored.")
        if not self.apply_pca:
            warnings.warn(
                "Running ANNOY Indexing on high-dimensional data. Nearest-neighbor search may be slow!")

    def decide_num_pairs(self, n):
        if self.n_neighbors is None:
            if n <= 10000:
                self.n_neighbors = 10
            else:
                self.n_neighbors = int(round(10 + 15 * (np.log10(n) - 4)))
        self.n_MN = int(round(self.n_neighbors * self.MN_ratio))
        self.n_FP = int(round(self.n_neighbors * self.FP_ratio))
        if self.n_neighbors < 1:
            raise ValueError(
                "The number of nearest neighbors can't be less than 1")
        if self.n_FP < 1:
            raise ValueError(
                "The number of further points can't be less than 1")

    def fit(self, X, init=None, save_pairs=True):
        '''Projects a high dimensional dataset into a low-dimensional embedding, without returning the output.

        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the PaCMAP instance.

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.

        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        '''

        X = np.copy(X).astype(np.float32)
        # Preprocess the dataset
        n, dim = X.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
        X, pca_solution, tsvd, self.xmin, self.xmax, self.xmean = preprocess_X(
            X, self.distance, self.apply_pca, self.verbose, self.random_state, dim, self.n_components)
        self.tsvd_transformer = tsvd
        self.pca_solution = pca_solution
        # Deciding the number of pairs
        self.decide_num_pairs(n)
        print_verbose(
            "PaCMAP(n_neighbors={}, n_MN={}, n_FP={}, distance={}, "
            "lr={}, n_iters={}, apply_pca={}, opt_method='adam', "
            "verbose={}, intermediate={}, seed={})".format(
                self.n_neighbors,
                self.n_MN,
                self.n_FP,
                self.distance,
                self.lr,
                self.num_iters,
                self.apply_pca,
                self.verbose,
                self.intermediate,
                _RANDOM_STATE
            ), self.verbose
        )
        # Sample pairs
        self.sample_pairs(X, self.save_tree)
        self.num_instances = X.shape[0]
        self.num_dimensions = X.shape[1]
        # Initialize and Optimize the embedding
        self.embedding_, self.intermediate_states, self.pair_neighbors, self.pair_MN, self.pair_FP = pacmap(
            X,
            self.n_components,
            self.pair_neighbors,
            self.pair_MN,
            self.pair_FP,
            self.lr,
            self.num_iters,
            init,
            self.verbose,
            self.intermediate,
            self.intermediate_snapshots,
            pca_solution,
            self.tsvd_transformer
        )
        if not save_pairs:
            self.del_pairs()
        return self

    def fit_transform(self, X, init=None, save_pairs=True):
        '''Projects a high dimensional dataset into a low-dimensional embedding and return the embedding.

        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the PaCMAP instance.

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset.
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.

        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        '''

        self.fit(X, init, save_pairs)
        if self.intermediate:
            return self.intermediate_states
        else:
            return self.embedding_

    def transform(self, X, basis=None, init=None, save_pairs=True):
        '''Projects a high dimensional dataset into existing embedding space and return the embedding.
        Warning: In the current version of implementation, the `transform` method will treat the input as an 
        additional dataset, which means the same point could be mapped into a different place.

        Parameters
        ---------
        X: numpy.ndarray
            The new high-dimensional dataset that is being projected. 
            An embedding will get created based on parameters of the PaCMAP instance.

        basis: numpy.ndarray
            The original dataset that have already been applied during the `fit` or `fit_transform` process.
            If `save_tree == False`, then the basis is required to reconstruct the ANNOY tree instance.
            If `save_tree == True`, then it's unnecessary to provide the original dataset again.

        init: str, optional
            One of ['pca', 'random']. Initialization of the embedding, default='pca'.
            If 'pca', then the low dimensional embedding is initialized to the PCA mapped dataset. 
            The PCA instance will be the same one that was applied to the original dataset during the `fit` or `fit_transform` process. 
            If 'random', then the low dimensional embedding is initialized with a Gaussian distribution.

        save_pairs: bool, optional
            Whether to save the pairs that are sampled from the dataset. Useful for reproducing results.
        '''

        # Preprocess the data
        X = np.copy(X).astype(np.float32)
        X = preprocess_X_new(X, self.distance, self.xmin, self.xmax,
                             self.xmean, self.tsvd_transformer,
                             self.apply_pca, self.verbose)
        if basis is not None and self.tree is None:
            basis = np.copy(basis).astype(np.float32)
            basis = preprocess_X_new(basis, self.distance, self.xmin, self.xmax,
                                     self.xmean, self.tsvd_transformer,
                                     self.apply_pca, self.verbose)
        # Sample pairs
        self.pair_XP = generate_extra_pair_basis(basis, X,
                                                 self.n_neighbors,
                                                 self.tree,
                                                 self.distance,
                                                 self.verbose
                                                 )
        if not save_pairs:
            self.pair_XP = None

        # Initialize and Optimize the embedding
        Y, intermediate_states = pacmap_fit(X, self.embedding_, self.n_components, self.pair_XP, self.lr,
                                            self.num_iters, init, self.verbose,
                                            self.intermediate, self.intermediate_snapshots,
                                            self.pca_solution, self.tsvd_transformer)
        if self.intermediate:
            return intermediate_states
        else:
            return Y[self.embedding_.shape[0]:, :]

    def sample_pairs(self, X, save_tree):
        '''
        Sample PaCMAP pairs from the dataset.

        Parameters
        ---------
        X: numpy.ndarray
            The high-dimensional dataset that is being projected.

        save_tree: bool
            Whether to save the annoy index tree after finding the nearest neighbor pairs.
        '''
        # Creating pairs
        print_verbose("Finding pairs", self.verbose)
        if self.pair_neighbors is None:
            self.pair_neighbors, self.pair_MN, self.pair_FP, self.tree = generate_pair(
                X, self.n_neighbors, self.n_MN, self.n_FP, self.distance, self.verbose
            )
            print_verbose("Pairs sampled successfully.", self.verbose)
        elif self.pair_MN is None and self.pair_FP is None:
            print_verbose(
                "Using user provided nearest neighbor pairs.", self.verbose)
            assert self.pair_neighbors.shape == (
                X.shape[0] * self.n_neighbors, 2), "The shape of the user provided nearest neighbor pairs is incorrect."
            self.pair_neighbors, self.pair_MN, self.pair_FP = generate_pair_no_neighbors(
                X, self.n_neighbors, self.n_MN, self.n_FP, self.pair_neighbors, self.distance, self.verbose
            )
            print_verbose("Pairs sampled successfully.", self.verbose)
        else:
            print_verbose("Using stored pairs.", self.verbose)

        if not save_tree:
            self.tree = None

        return self

    def del_pairs(self):
        '''
        Delete stored pairs.
        '''
        self.pair_neighbors = None
        self.pair_MN = None
        self.pair_FP = None
        return self
