import numpy as np
from data import data_prepare
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from collections import Counter
from numpy.random import default_rng


def knn_clf(nbr_vec, y):
    '''
    Helper function to evaluate the knn accuracy
    '''
    y_vec = y[nbr_vec]
    c = Counter(y_vec)
    return c.most_common(1)[0][0]


def knn_eval(X, y, n_neighbors=1):
    '''
    Helper function to evaluate the knn accuracy.
    '''
    sum_acc = 0
    max_acc = X.shape[0]
    # Train once, reuse multiple times
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    for i in range(X.shape[0]):
        result = knn_clf(indices[i], y)
        if result == y[i]:
            sum_acc += 1
    avg_acc = sum_acc / max_acc
    return avg_acc


def knn_eval_series(X, y, n_neighbors_list=[1, 3, 5, 10, 15, 20, 25, 30]):
    '''
    Evaluate the knn accuracy.
    '''
    avg_accs = []
    for n_neighbors in n_neighbors_list:
        avg_acc = knn_eval(X, y, n_neighbors)
        avg_accs.append(avg_acc)
    return avg_accs


def faster_knn_eval_series(X, y, n_neighbors_list=[1, 3, 5, 10, 15, 20, 25, 30]):
    '''
    Same algorithm as above, but faster.
    '''
    avg_accs = []
    max_acc = X.shape[0]
    # Train once, reuse multiple times
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_list[-1] + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    for n_neighbors in n_neighbors_list:
        sum_acc = 0
        for i in range(X.shape[0]):
            indices_temp = indices[:, :n_neighbors]
            result = knn_clf(indices_temp[i], y)
            if result == y[i]:
                sum_acc += 1
        avg_acc = sum_acc / max_acc
        avg_accs.append(avg_acc)
    return avg_accs


def svm_eval(X, y, img_verbose=False, n_splits=5, **kwargs):
    '''
    Ten fold SVM accuracy evaluation.
    '''
    X = scale(X)
    skf = StratifiedKFold(n_splits=n_splits)
    sum_acc = 0
    max_acc = n_splits
    for train_index, test_index in skf.split(X, y):
        clf = SVC(**kwargs)
        clf.fit(X[train_index], y[train_index])
        acc = clf.score(X[test_index], y[test_index])
        sum_acc += acc
    avg_acc = sum_acc / max_acc
    return avg_acc


def faster_svm_eval(X, y, n_splits=5, **kwargs):
    '''
    Save algorithm as above, but faster by approximating kernels.
    '''
    X = X.astype(np.float)
    X = scale(X)
    skf = StratifiedKFold(n_splits=n_splits)
    sum_acc = 0
    max_acc = n_splits
    for train_index, test_index in skf.split(X, y):
        feature_map_nystroem = Nystroem(gamma=1 / (X.var() * X.shape[1]), random_state=1, n_components=300)
        data_transformed = feature_map_nystroem.fit_transform(X[train_index])
        clf = LinearSVC(random_state=0, tol=1e-5, **kwargs)
        clf.fit(data_transformed, y[train_index])
        test_transformed = feature_map_nystroem.transform(X[test_index])
        acc = clf.score(test_transformed, y[test_index])
        sum_acc += acc
    avg_acc = sum_acc / max_acc
    return avg_acc


def centroid_triplet_eval(X, X_new, y):
    '''
    Centroid is defined by the mass center of each cluster.
    '''
    cluster_mean_ori, cluster_mean_new = [], []
    categories = np.unique(y)
    num_cat = len(categories)
    mask = np.mask_indices(num_cat, np.tril, -1)
    for i in range(num_cat):
        label = categories[i]
        X_clus_ori = X[y == label]
        X_clus_new = X_new[y == label]
        cluster_mean_ori.append(np.mean(X_clus_ori, axis=0))
        cluster_mean_new.append(np.mean(X_clus_new, axis=0))
    cluster_mean_ori = np.array(cluster_mean_ori)
    cluster_mean_new = np.array(cluster_mean_new)
    ori_dist = euclidean_distances(cluster_mean_ori)[mask]
    new_dist = euclidean_distances(cluster_mean_new)[mask]
    dist_agree = 0.  # two distance agrees
    dist_all = 0.  # count
    for i in range(len(ori_dist)):
        for j in range(i + 1, len(ori_dist)):
            if ori_dist[i] > ori_dist[j] and new_dist[i] > new_dist[j]:
                dist_agree += 1
            elif ori_dist[i] <= ori_dist[j] and new_dist[i] <= new_dist[j]:
                dist_agree += 1
            dist_all += 1
    return dist_agree / dist_all


def random_triplet_eval(X, X_new, y):
    '''
    Input X should be a PCA-transformed original dataset.
    '''

    # Sampling Triplets
    # Five triplet per point
    anchors = np.arange(X.shape[0])
    rng = default_rng()
    triplets = rng.choice(anchors, (X.shape[0], 5, 2))
    triplet_labels = np.zeros((X.shape[0], 5))
    anchors = anchors.reshape((-1, 1, 1))

    # Calculate the distances and generate labels
    b = np.broadcast(anchors, triplets)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u, v) in b]
    labels = distances[:, :, 0] < distances[:, :, 1]

    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u, v) in b]
    pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]
    correct = np.sum(pred_vals == labels)
    acc = correct / X.shape[0] / 5
    return acc