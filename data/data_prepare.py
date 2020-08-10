from sklearn.datasets import make_swiss_roll, make_s_curve
import tensorflow.compat.v1 as tf
import numpy as np

def data_prep(dataset='MNIST', size=10000):
    if dataset == 'MNIST':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        X = np.concatenate((x_train, x_test), axis=0).reshape(70000, 28*28).astype(np.float32)
        labels = np.concatenate((y_train, y_test))
    elif dataset == 'F-MNIST':
        fmnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fmnist.load_data()
        X = np.concatenate((x_train, x_test), axis=0).reshape(-1, 28*28).astype(np.float32)
        labels = np.concatenate((y_train, y_test))
    elif dataset == 'cifar-10':
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        X = np.concatenate((x_train, x_test), axis=0).reshape(60000, 32*32*3).astype(np.float32)
        labels = np.concatenate((y_train, y_test)).reshape(60000)
    elif dataset == 'swiss_roll':
        X, labels = make_swiss_roll(n_samples=size)
    elif dataset == 's_curve':
        X, labels = make_s_curve(n_samples=size)
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