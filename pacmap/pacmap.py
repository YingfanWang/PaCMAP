import numpy as np
from sklearn.base import BaseEstimator
import numba
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time
import math
import datetime
import warnings


@numba.njit("f4(f4[:])")
def l2_norm(x):
    """
    L2 norm of a vector.
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])")
def euclid_dist(x1, x2):
    """
    Euclidean distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)


@numba.njit("f4(f4[:],f4[:])")
def manhattan_dist(x1, x2):
    """
    Manhattan distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += np.abs(x1[i] - x2[i])
    return result


@numba.njit("f4(f4[:],f4[:])")
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


@numba.njit("f4(f4[:],f4[:])")
def hamming_dist(x1, x2):
    """
    Hamming distance between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        if x1[i] != x2[i]:
            result += 1.0
    return result


@numba.njit()
def calculate_dist(x1, x2, distance_index):
    if distance_index == 0:
        return euclid_dist(x1, x2)
    elif distance_index == 1:
        return manhattan_dist(x1, x2)
    elif distance_index == 2:
        return angular_dist(x1, x2)
    elif distance_index == 3:
        return hamming_dist(x1, x2)


@numba.njit("i4[:](i4,i4,i4[:])", nogil=True)
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


@numba.njit("i4[:,:](f4[:,:],f4[:,:],i4[:,:],i4)", parallel=True, nogil=True)
def sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors):
    n = X.shape[0]
    pair_neighbors = np.empty((n*n_neighbors, 2), dtype=np.int32)

    for i in numba.prange(n):
        scaled_sort = np.argsort(scaled_dist[i])
        for j in numba.prange(n_neighbors):
            pair_neighbors[i*n_neighbors + j][0] = i
            pair_neighbors[i*n_neighbors + j][1] = nbrs[i][scaled_sort[j]]
    return pair_neighbors


@numba.njit("i4[:,:](f4[:,:],i4)", nogil=True)
def sample_MN_pair(X, n_MN):
    n = X.shape[0]
    pair_MN = np.empty((n*n_MN, 2), dtype=np.int32)
    for i in numba.prange(n):
        for jj in range(n_MN):
            sampled = np.random.randint(0, n, 6)
            dist_list = np.empty((6), dtype=np.float32)
            for t in range(sampled.shape[0]):
                dist_list[t] = euclid_dist(X[i], X[sampled[t]])
            min_dic = np.argmin(dist_list)
            dist_list = np.delete(dist_list, [min_dic])
            sampled = np.delete(sampled, [min_dic])
            picked = sampled[np.argmin(dist_list)]
            pair_MN[i*n_MN + jj][0] = i
            pair_MN[i*n_MN + jj][1] = picked
    return pair_MN


@numba.njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True)
def sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP):
    n = X.shape[0]
    pair_FP = np.empty((n * n_FP, 2), dtype=np.int32)
    for i in numba.prange(n):
        for k in numba.prange(n_FP):
            FP_index = sample_FP(
                n_FP, n, pair_neighbors[i*n_neighbors: i*n_neighbors + n_neighbors][1])
            pair_FP[i*n_FP + k][0] = i
            pair_FP[i*n_FP + k][1] = FP_index[k]
    return pair_FP


@numba.njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True)
def scale_dist(knn_distance, sig, nbrs):
    n, num_neighbors = knn_distance.shape
    scaled_dist = np.zeros((n, num_neighbors), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(num_neighbors):
            scaled_dist[i, j] = knn_distance[i, j] ** 2 / \
                sig[i] / sig[nbrs[i, j]]
    return scaled_dist


@numba.njit("void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,f4,f4,i4)", parallel=True, nogil=True)
def update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr):
    n, dim = Y.shape
    lr_t = lr * math.sqrt(1.0 - beta2**(itr+1)) / (1.0 - beta1**(itr+1))
    for i in numba.prange(n):
        for d in numba.prange(dim):
            m[i][d] += (1 - beta1) * (grad[i][d] - m[i][d])
            v[i][d] += (1 - beta2) * (grad[i][d]**2 - v[i][d])
            Y[i][d] -= lr_t * m[i][d]/(math.sqrt(v[i][d]) + 1e-7)


@numba.njit("f4[:,:](f4[:,:],i4[:,:],i4[:,:],i4[:,:],f4,f4,f4)", parallel=True, nogil=True)
def pacmap_grad(Y, pair_neighbors, pair_MN, pair_FP, w_neighbors, w_MN, w_FP):
    n, dim = Y.shape
    grad = np.zeros((n+1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    loss = np.zeros(3, dtype=np.float32)
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


def generate_pair(
        X,
        n_neighbors,
        n_MN,
        n_FP,
        distance='euclidean',
        verbose=True
):
    n, dim = X.shape
    n_neighbors_extra = min(n_neighbors + 50, n)
    tree = AnnoyIndex(dim, metric=distance)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(20)

    nbrs = np.zeros((n, n_neighbors_extra), dtype=np.int32)
    knn_distances = np.empty((n, n_neighbors_extra), dtype=np.float32)

    for i in range(n):
        nbrs_ = tree.get_nns_by_item(i, n_neighbors_extra+1)
        nbrs[i, :] = nbrs_[1:]
        for j in range(n_neighbors_extra):
            knn_distances[i, j] = tree.get_distance(i, nbrs[i, j])
    if verbose:
        print("Found nearest neighbor")
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)
    if verbose:
        print("Found sig")
    scaled_dist = scale_dist(knn_distances, sig, nbrs)
    if verbose:
        print("Found scaled dist")
    pair_neighbors = sample_neighbors_pair(X, scaled_dist, nbrs, n_neighbors)
    pair_MN = sample_MN_pair(X, n_MN)
    pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    return pair_neighbors, pair_MN, pair_FP

def generate_pair_no_neighbors(
        X,
        n_neighbors,
        n_MN,
        n_FP,
        pair_neighbors,
        distance='euclidean',
        verbose=True
):
    pair_MN = sample_MN_pair(X, n_MN)
    pair_FP = sample_FP_pair(X, pair_neighbors, n_neighbors, n_FP)
    return pair_neighbors, pair_MN, pair_FP


def pacmap(
        X,
        n_dims,
        n_neighbors,
        n_MN,
        n_FP,
        pair_neighbors,
        pair_MN,
        pair_FP,
        distance,
        lr,
        num_iters,
        Yinit,
        apply_pca,
        verbose,
        intermediate
):
    start_time = time.time()
    n, high_dim = X.shape

    if intermediate:
        itr_dic = [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450]
        intermediate_states = np.empty((13, n, 2), dtype=np.float32)
    else:
        intermediate_states = None

    pca_solution = False
    if pair_neighbors is None:
        if verbose:
            print("Finding pairs!")
        if distance != "hamming":
            if high_dim > 100 and apply_pca:
                X -= np.mean(X, axis=0)
                X = TruncatedSVD(n_components=100,
                                 random_state=0).fit_transform(X)
                pca_solution = True
                if verbose:
                    print("Applied PCA, the dimensionality becomes 100")
            else:
                X -= np.min(X)
                X /= np.max(X)
                X -= np.mean(X, axis=0)
                if verbose:
                    print(X)
        pair_neighbors, pair_MN, pair_FP = generate_pair(
            X, n_neighbors, n_MN, n_FP, distance, verbose
        )
        if verbose:
            print("Pairs sampled successfully.")
    elif pair_MN is None and pair_FP is None:
        if verbose:
            print("Using user provided nearest neighbor pairs.")
        try:
            assert(pair_neighbors.shape == (n * n_neighbors, 2))
        except AssertionError:
            print("The shape of the user provided nearest neighbor pairs is incorrect.")
            raise ValueError
        pair_neighbors, pair_MN, pair_FP = generate_pair_no_neighbors(
            X, n_neighbors, n_MN, n_FP, pair_neighbors, distance, verbose
        )
        if verbose:
            print("Pairs sampled successfully.")
    else:
        if verbose:
            print("Using stored pairs.")

    if Yinit is None or Yinit == "pca":
        if pca_solution:
            Y = 0.01 * X[:, :n_dims]
        else:
            Y = 0.01 * \
                PCA(n_components=n_dims).fit_transform(X).astype(np.float32)
    elif Yinit == "random":
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    else:  # user_supplied matrix
        Yinit = Yinit.astype(np.float32)
        scaler = preprocessing.StandardScaler().fit(Yinit)
        Y = scaler.transform(Yinit) * 0.0001

    w_MN_init = 1000.
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(Y, dtype=np.float32)
    v = np.zeros_like(Y, dtype=np.float32)

    if intermediate:
        itr_ind = 1
        intermediate_states[0, :, :] = Y

    for itr in range(num_iters):
        if itr < 100:
            w_MN = (1 - itr/100) * w_MN_init + itr/100 * 3.0
            w_neighbors = 2.0
            w_FP = 1.0
        elif itr < 200:
            w_MN = 3.0
            w_neighbors = 3
            w_FP = 1
        else:
            w_MN = 0.0
            w_neighbors = 1.
            w_FP = 1.

        grad = pacmap_grad(Y, pair_neighbors, pair_MN,
                           pair_FP, w_neighbors, w_MN, w_FP)
        C = grad[-1, 0]
        update_embedding_adam(Y, grad, m, v, beta1, beta2, lr, itr)

        if intermediate:
            if (itr+1) == itr_dic[itr_ind]:
                intermediate_states[itr_ind, :, :] = Y
                itr_ind += 1
                if itr_ind > 12:
                    itr_ind -= 1
        if verbose:
            if (itr + 1) % 10 == 0:
                print("Iteration: %4d, Loss: %f" % (itr + 1, C))

    if verbose:
        elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
        print("Elapsed time: %s" % (elapsed))
    return Y, intermediate_states, pair_neighbors, pair_MN, pair_FP


class PaCMAP(BaseEstimator):
    def __init__(self,
                 n_dims=2,
                 n_neighbors=10,
                 MN_ratio=0.5,
                 FP_ratio=2.0,
                 pair_neighbors=None,
                 pair_MN=None,
                 pair_FP=None,
                 distance="euclidean",
                 lr=1.0,
                 num_iters=450,
                 verbose=False,
                 apply_pca=True,
                 intermediate=False
                 ):
        self.n_dims = n_dims
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.pair_neighbors = pair_neighbors
        self.pair_MN = pair_MN
        self.pair_FP = pair_FP
        self.distance = distance
        self.lr = lr
        self.num_iters = num_iters
        self.apply_pca = apply_pca
        self.verbose = verbose
        self.intermediate = intermediate

        if self.n_dims < 2:
            raise ValueError(
                "The number of projection dimensions must be at least 2")
        if self.lr <= 0:
            raise ValueError("The learning rate must be larger than 0")
        if self.distance == "hamming" and apply_pca:
            warnings.warn("apply_pca = True for Hamming distance.")
        if not self.apply_pca:
            print(
                "running ANNOY on high-dimensional data. nearest-neighbor search may be slow!")

    def fit(self, X, init=None, save_pairs=True):
        X = X.astype(np.float32)
        n, dim = X.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
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
        if self.verbose:
            print(
                "PaCMAP(n_neighbors={}, n_MN={}, n_FP={}, distance={},"
                "lr={}, n_iters={}, apply_pca={}, opt_method='adam', verbose={}, intermediate={})".format(
                    self.n_neighbors,
                    self.n_MN,
                    self.n_FP,
                    self.distance,
                    self.lr,
                    self.num_iters,
                    self.apply_pca,
                    self.verbose,
                    self.intermediate
                )
            )
        if save_pairs:
            self.embedding_, self.intermediate_states, self.pair_neighbors, self.pair_MN, self.pair_FP = pacmap(
                X,
                self.n_dims,
                self.n_neighbors,
                self.n_MN,
                self.n_FP,
                self.pair_neighbors,
                self.pair_MN,
                self.pair_FP,
                self.distance,
                self.lr,
                self.num_iters,
                init,
                self.apply_pca,
                self.verbose,
                self.intermediate
            )
        else:
            self.embedding_, self.intermediate_states, _, _, _ = pacmap(
                X,
                self.n_dims,
                self.n_neighbors,
                self.n_MN,
                self.n_FP,
                self.pair_neighbors,
                self.pair_MN,
                self.pair_FP,
                self.distance,
                self.lr,
                self.num_iters,
                init,
                self.apply_pca,
                self.verbose,
                self.intermediate
            )

        return self

    def fit_transform(self, X, init="random", save_pairs=True):
        self.fit(X, init, save_pairs)
        if self.intermediate:
            return self.intermediate_states
        else:
            return self.embedding_

    def sample_pairs(self, X):
        if self.verbose:
            print("sampling pairs")
        X = X.astype(np.float32)
        n, dim = X.shape
        if n <= 0:
            raise ValueError("The sample size must be larger than 0")
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
        if self.distance != "hamming":
            if X.shape[1] > 100 and self.apply_pca:
                X -= np.mean(X, axis=0)
                X = TruncatedSVD(n_components=100,
                                 random_state=0).fit_transform(X)
                if self.verbose:
                    print("applied PCA")
            else:
                X -= np.min(X)
                X /= np.max(X)
                X -= np.mean(X, axis=0)
        self.pair_neighbors, self.pair_MN, self.pair_FP = generate_pair(
            X,
            self.n_neighbors,
            self.n_MN,
            self.n_FP,
            self.distance,
            self.verbose
        )
        if self.verbose:
            print("sampled pairs")

        return self

    def del_pairs(self):
        self.pair_neighbors = None,
        self.pair_MN = None,
        self.pair_FP = None,
        return self
