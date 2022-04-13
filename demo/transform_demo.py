'''
A script that demonstrates the transform feature of pacmap.
The MNIST dataset will be separated into n-folds, where (n-1) folds will be used
to fit a PaCMAP instance, and the last fold will be used as a test set.
We use the transform feature to map the test set into the already constructed
embedding space.
'''

import pacmap
import numpy as np
from sklearn.model_selection import StratifiedKFold
from demo_utils import *


# MNIST
# You may want to unzip the mnist_images.npy.zip before doing this step
mnist = np.load("../data/mnist_images.npy", allow_pickle=True)
mnist = mnist.reshape(mnist.shape[0], -1)
labels = np.load("../data/mnist_labels.npy", allow_pickle=True)
print("Loading data")
n_splits = [2, 5, 10]
for n in n_splits:
    skf = StratifiedKFold(n_splits=n)
    for train_index, test_index in skf.split(mnist, labels):
        X_train, X_test = mnist[train_index], mnist[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        break
    
    # Initialize the instance
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=20, save_tree=False)

    # Fit the training set
    embedding = reducer.fit_transform(X_train)

    # Transform the test set into the same embedding space
    embedding_test = reducer.transform(X_test, basis=X_train)

    # Plot the results
    embedding_combined = np.concatenate((embedding, embedding_test))
    y = np.concatenate((y_train, y_test))
    embeddings = [embedding, embedding_test, embedding_combined]
    labelset = [y_train, y_test, y]
    titles = ['Training', 'Test', 'Combined']
    generate_combined_figure(embeddings, labelset, titles, f'mnist_transform_{n}')
