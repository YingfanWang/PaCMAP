import FlowCal
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold, datasets
from time import time
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll, make_s_curve


def data_prep(data_path, dataset='MNIST', size=10000):
    '''
    This function loads the dataset as numpy array.
    Input:
        data_path: path of the folder you store all the data needed.
        dataset: the name of the dataset.
        size: the size of the dataset. This is useful when you only
              want to pick a subset of the data
    Output:
        X: the dataset in numpy array
        labels: the labels of the dataset.
    '''
    if dataset == 'MNIST':
        X = np.load(data_path + '/mnist_images.npy', allow_pickle=True).reshape(70000, 28*28)
        labels = np.load(data_path + '/mnist_labels.npy', allow_pickle=True)
    elif dataset == 'FMNIST':
        X = np.load(data_path + '/fmnist_images.npy', allow_pickle=True).reshape(70000, 28*28)
        labels = np.load(data_path + '/fmnist_labels.npy', allow_pickle=True)
    elif dataset == 'coil_20':
        X = np.load(data_path + '/coil_20.npy', allow_pickle=True).reshape(1440, 128*128)
        labels = np.load(data_path + '/coil_20_labels.npy', allow_pickle=True)
    elif dataset == 'coil_100':
        X = np.load(data_path + '/coil_100.npy', allow_pickle=True).reshape(7200, -1)
        labels = np.load(data_path + '/usr/xtmp/hyhuang/MNIST/coil_100_labels.npy', allow_pickle=True)
    elif dataset == 'mammoth':
        with open(data_path + '/mammoth_3d.json', 'r') as f:
            X = json.load(f)
        X = np.array(X)
        with open(data_path + '/mammoth_umap.json', 'r') as f:
            labels = json.load(f)
        labels = labels['labels']
        labels = np.array(labels)
    elif dataset == 'mammoth_50k':
        with open(data_path + '/mammoth_3d_50k.json', 'r') as f:
            X = json.load(f)
        X = np.array(X)
        labels = np.zeros(10)
    elif dataset == 'Flow_cytometry':
        X = FlowCal.io.FCSData(data_path + '/11-12-15_314.fcs')
        labels = np.zeros(10)
    elif dataset == 'Mouse_scRNA':
        data = pd.read_csv(data_path + '/GSE93374_Merged_all_020816_BatchCorrected_LNtransformed_doubletsremoved_Data.txt', sep='\t')
        X = data.to_numpy()
        labels = pd.read_csv(data_path + '/GSE93374_cell_metadata.txt', sep='\t')
    elif dataset == 'swiss_roll':
        X, labels = make_swiss_roll(n_samples=size, random_state=20200202)
    elif dataset == 's_curve':
        X, labels = make_s_curve(n_samples=size, random_state=20200202)
    elif dataset == 's_curve_hole':
        X, labels = make_s_curve(n_samples=size, random_state=20200202)
        anchor = np.array([0, 1, 0])
        indices = np.sum(np.square(X-anchor), axis=1) > 0.3
        X, labels = X[indices], labels[indices]
    elif dataset == 'swiss_roll_hole':
        X, labels = make_swiss_roll(n_samples=size, random_state=20200202)
        anchor = np.array([-10, 10, 0])
        indices = np.sum(np.square(X-anchor), axis=1) > 20
        X, labels = X[indices], labels[indices]
    elif dataset == 'kddcup99':
        X = np.load(data_path + '/KDDcup99_float.npy', allow_pickle=True)
        labels = np.load(data_path + '/KDDcup99_labels_int.npy', allow_pickle=True)
    elif dataset == '20NG':
        X = np.load(data_path + '/20NG.npy', allow_pickle=True)
        labels = np.load(data_path + '/20NG_labels.npy', allow_pickle=True)
    elif dataset == 'USPS':
        X = np.load(data_path + '/USPS.npy', allow_pickle=True)
        labels = np.load(data_path + '/USPS_labels.npy', allow_pickle=True)
    elif dataset == 'cifar10':
        X = np.load(data_path + '/cifar10_imgs.npy', allow_pickle=True)
        labels = np.load('/cifar10_labels.npy', allow_pickle=True)
    elif dataset == 'cifar100':
        X = np.load(data_path + '/cifar100_imgs.npy', allow_pickle=True)
        labels = np.load('/cifar100_labels.npy', allow_pickle=True)
    else:
        print('Unsupported dataset')
        assert(False)
    return X[:size], labels[:size]

def experiment(X, method='PaCMAP', **kwargs):
    if method == 'PaCMAP':
        transformer = PaCMAP(**kwargs)
    elif method == 'UMAP':
        transformer = umap.UMAP(**kwargs)
    elif method == 'TriMAP':
        transformer = trimap.TRIMAP(**kwargs)
    elif method == 'LargeVis':
        transformer = LargeVis(**kwargs)
    elif method == 't-SNE':
        transformer = TSNE(**kwargs)
    else:
        print("Incorrect method specified")
        assert(False)
    start_time = time()
    X_low = transformer.fit_transform(X)
    total_time = time() - start_time
    print("This run's time:")
    print(total_time)
    return X_low, total_time

def experiment_five(X, method='PaCMAP', **kwargs):
    length = X.shape[0]
    X_lows, all_times = [], []
    for i in range(5):
        X_low, all_time = experiment(X, method, **kwargs)
        X_lows.append(X_low)
        all_times.append(all_time)
    X_lows = np.array(X_lows)
    all_times = np.array(all_times)
    return X_lows, all_times

def main(data_path, output_path, dataset_name='MNIST', size=10000000):
    X, labels = data_prep(data_path, dataset=dataset_name, size=size)
    if dataset_name == 'Mouse_scRNA':
        pca = PCA(n_components=1000)
        X = pca.fit_transform(X)
    elif X.shape[1] > 100:
        pca = PCA(n_components=100)
        X = pca.fit_transform(X)
    print("Data loaded successfully")
    methods = ['t-SNE']
    args = {'t-SNE':[{'perplexity':10}, {'perplexity':20}, {'perplexity':40}]}
    print("Experiment started")
    for method in methods:
        parameters = args[method]
        for parameter in parameters:
            X_low, total_time = experiment_five(X, method, **parameter)
            if 'n_neighbors' in parameter:
                n_neighbors = parameter['n_neighbors']
            elif 'perplexity' in parameter:
                n_neighbors = parameter['perplexity']
            else:
                n_neighbors = 10 # Default value
            loc_string = output_path + \
                         '{dataset_name}_{method}_{n_neighbors}'.format(dataset_name=dataset_name, method=method, n_neighbors=n_neighbors)
            np.save(loc_string, X_low)
            avg_time = np.mean(total_time)
            print('Average time for method {method} on {dataset_name} with param={n_neighbors} is {avg_time}'.format(dataset_name=dataset_name, method=method, n_neighbors=n_neighbors, avg_time=avg_time))
            print('The detailed time is {total_time}'.format(total_time=total_time))
    return 0

if __name__ == '__main__':
    # Please define the data_path and output_path here
    data_path = "../data/"
    output_path = "../output/"
    main(data_path, output_path, 'MNIST')
    main(data_path, output_path, 'FMNIST')
    main(data_path, output_path, 'coil_20')
    main(data_path, output_path, 'coil_100')
    main(data_path, output_path, 'Mouse_scRNA')
    main(data_path, output_path, 'mammoth')
    main(data_path, output_path, 's_curve', 10000)
    main(data_path, output_path, 's_curve_hole', 10000)
    main(data_path, output_path, '20NG', 100000)
    main(data_path, output_path, 'USPS', 100000)
    main(data_path, output_path, 'kddcup99', 10000000)
    main(data_path, output_path, 'cifar10', 10000000)
    main(data_path, output_path, 'cifar100', 10000000)