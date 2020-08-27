import FlowCal
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import LargeVis

from sklearn import manifold, datasets
from time import time
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll, make_s_curve


def data_path_finder(data_path, dataset_name='MNIST'):
    if dataset_name == 'MNIST':
        input_file = data_path + '/mnist_images.txt'
    elif dataset_name == 'FMNIST':
        input_file = data_path + '/fmnist_images.txt'
    elif dataset_name == 'coil_20':
        input_file = '/usr/xtmp/hyhuang/LargeVisInputs/coil_20.txt'
    elif dataset_name == 'coil_100':
        input_file = data_path + '/coil_100.txt'
    elif dataset_name == 'Flow_cytometry':
        input_file = data_path + '/opt_fcs.txt'
    elif dataset_name == 'Mouse_scRNA':
        input_file = data_path + '/Mouse_RNA.txt'
    elif dataset_name == 'mammoth':
        input_file = data_path + '/mammoth_3d.txt'
    elif dataset_name == 'mammoth_50k':
        input_file = data_path + '/mammoth_3d_50k.txt'
    elif dataset_name == 's_curve':
        input_file = data_path + '/s_curve.txt'
    elif dataset_name == 's_curve_hole':
        input_file = data_path + '/s_curve_hole.txt'
    elif dataset_name == '20NG':
        input_file = data_path + '/20NG.txt'
    elif dataset_name == 'USPS':
        input_file = data_path + '/USPS.txt'
    elif dataset_name == 'kddcup99':
        input_file = data_path + '/KDDcup99.txt'
    elif dataset_name == 'cifar10':
        input_file = data_path + '/cifar10.txt'
    elif dataset_name == 'cifar100':
        input_file = data_path + '/cifar100.txt'
    return input_file


def experiment_five(data_path, output_path, dataset_name='MNIST', n_neighbors=80):
    X_lows, total_times = [], []
    for i in range(5):
        input_path = data_path_finder(data_path, dataset_name)
        LargeVis.loadfile(input_path)
        start_time = time()
        X_low = LargeVis.run(2, 16, -1, -1, -1, -1, -1, 3*n_neighbors, -1, n_neighbors) # -1 means default value
        total_time = time() - start_time
        method = 'LargeVis'
        X_lows.append(X_low)
        total_times.append(total_time)
    X_lows = np.array(X_lows)
    total_times = np.array(total_times)
    avg_time = np.mean(total_times)
    np.save(output_path + '/{dataset_name}_{method}_{n_neighbors}'.format(dataset_name\
            =dataset_name,method=method,n_neighbors=n_neighbors), X_lows)
    print('Total time for method {method} on {dataset_name} with {n_neighbors} is {avg_time}'.format(method=method, dataset_name=dataset_name, avg_time=avg_time, n_neighbors=n_neighbors))
    print('Detailed time is {total_times}'.format(total_times=total_times))


def main():
    dataset_name = 'mammoth'
    method = 'LargeVis'
    n_neighbors = 125
    input_path = data_path_finder(dataset_name)
    LargeVis.loadfile(input_path)
    start_time = time()
    X_low = LargeVis.run(2, 16, -1, -1, -1, -1, -1, 3*n_neighbors, -1, n_neighbors) # -1 means default value
    total_time = time() - start_time
    method = 'LargeVis'
    np.save('/home/home1/hh219/PaCMAP/output_{dataset_name}_{method}_{n_neighbors}'.format(dataset_name\
        =dataset_name,method=method,n_neighbors=n_neighbors), X_low)
    print('Total time for method {method} on {dataset_name} is {total_time}'.format(method=method, dataset_name=dataset_name, total_time=total_time))


if __name__ == '__main__':
    # Please define the data_path and output_path here
    data_path = "./data/"
    output_path = "./output/"
    experiment_five(data_path, output_path, 'MNIST')
    experiment_five(data_path, output_path, 'coil_20')
    experiment_five(data_path, output_path, 'FMNIST')
    experiment_five(data_path, output_path, 'coil_100')
    experiment_five(data_path, output_path, 'Mouse_scRNA')
    experiment_five(data_path, output_path, 'Flow_cytometry')
    experiment_five(data_path, output_path, 'mammoth')
    experiment_five(data_path, output_path, 's_curve')
    experiment_five(data_path, output_path, 's_curve_hole')
    experiment_five(data_path, output_path, 'kddcup99')
    experiment_five(data_path, output_path, '20NG')
    experiment_five(data_path, output_path, 'USPS')
    experiment_five(data_path, output_path, 'cifar10')
    experiment_five(data_path, output_path, 'cifar100')

