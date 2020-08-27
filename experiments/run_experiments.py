import umap
import trimap
import FlowCal
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import manifold, datasets
from time import time
from tqdm import tqdm
from sklearn.decomposition import PCA
from PaCMAP import PaCMAP
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
        labels = np.load(data_path + '/coil_100_labels.npy', allow_pickle=True)
    elif dataset == 'Flow_cytometry':
        X = FlowCal.io.FCSData(data_path + '/11-12-15_314.fcs')
        X = np.array(X)
        labels = np.zeros(10)
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
        labels = np.load(data_path + '/cifar10_labels.npy', allow_pickle=True)
    elif dataset == 'cifar100':
        X = np.load(data_path + '/cifar100_imgs.npy', allow_pickle=True)
        labels = np.load(data_path + '/cifar100_labels.npy', allow_pickle=True)
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
    elif dataset == '2D_curve':
        x = np.arange(-5.5, 9, 0.01)
        y = 0.01 * (x + 5) * (x + 2) * (x - 2) * (x - 6) * (x - 8)
        noise = np.random.randn(x.shape[0]) * 0.01
        y += noise
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        X = np.hstack((x, y))
        labels = x
    else:
        print('Unsupported dataset')
        assert(False)
    return X[:size], labels[:size]

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

def experiment(X, method='PaCMAP', **kwargs):
    if method == 'PaCMAP':
        transformer = PaCMAP(**kwargs)
    elif method == 'UMAP':
        transformer = umap.UMAP(**kwargs)
    elif method == 'TriMAP':
        transformer = trimap.TRIMAP(**kwargs)
    else:
        print("Incorrect method specified")
        assert(False)
    start_time = time()
    X_low = transformer.fit_transform(X)
    total_time = time() - start_time
    return X_low, total_time


def main(data_path, output_path, dataset_name = 'MNIST', size=10000, data_pca=True):
    X, labels = data_prep(dataset=dataset_name, size=size)
    if data_pca:
        if dataset_name == 'Mouse_scRNA':
            pca = PCA(n_components=1000)
            X = pca.fit_transform(X)
        elif X.shape[1] > 100:
            pca = PCA(n_components=100)
            X = pca.fit_transform(X)
    print("Data loaded successfully")
    # do experiment
    methods = ['PaCMAP', 'UMAP', 'TriMAP']

    args = {'TriMAP':[{'n_inliers':20}],
        'UMAP':[{'n_neighbors':10}, {'n_neighbors':20}, {'n_neighbors':40}],
        'PaCMAP':[{'num_NN':5, 'num_FP': 10, 'num_mid': 2},
        {'num_NN':10, 'num_FP': 20, 'num_mid': 5}, 
        {'num_NN':20, 'num_FP': 40, 'num_mid': 10}]}

    print("Experiment started")
    for method in methods:
        parameters = args[method]
        for parameter in parameters:
            X_low, total_time = experiment_five(X, method, **parameter)
            if 'n_neighbors' in parameter:
                n_neighbors = parameter['n_neighbors']
            elif 'n_inliers' in parameter:
                n_neighbors = parameter['n_inliers']
            elif 'num_NN' in parameter:
                n_neighbors = parameter['num_NN']
            elif 'perplexity' in parameter:
                n_neighbors = parameter['perplexity']
            else:
                n_neighbors = 'default' # Default value
            np.save(f'/home/home1/hh219/PaCMAP/output_5/{dataset_name}_{method}_{n_neighbors}', X_low)
            avg_time = np.mean(total_time)
            print(f'Average time for method {method} on {dataset_name} with param={n_neighbors} is {avg_time}')
            print(f'The detailed time is {total_time}')
    return 0

if __name__ == '__main__':
    # Please define the data_path and output_path here
    data_path = "./data/"
    output_path = "./output/"
    main(data_path, output_path,'MNIST', 10000000)
    main(data_path, output_path,'FMNIST', 10000000)
    main(data_path, output_path,'coil_20', 10000000)
    main(data_path, output_path,'coil_100', 10000000)

    main(data_path, output_path,'Mouse_scRNA', 10000000)
    main(data_path, output_path,'mammoth', 10000000)
    main(data_path, output_path,'s_curve_hole', 10000)
    main(data_path, output_path,'20NG', 20000)
    main(data_path, output_path,'USPS', 20000)


    main(data_path, output_path,'Flow_cytometry', 10000000)
    main(data_path, output_path,'kddcup99', 10000000)
