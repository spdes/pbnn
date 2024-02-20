import jax.random
import numpy as np
from sklearn.datasets import make_moons
from numpy.random import RandomState
from pbnn.data import DataSet
from pbnn.typings import JKey


class Moons(DataSet):

    def __init__(self, n: int, noise: float = 0.3, rng_state: RandomState = RandomState(666)):
        data = make_moons(n_samples=n, shuffle=True, noise=noise, random_state=rng_state)

        self.n = n
        self.xs = data[0]
        self.ys = self.reshape(data[1])

        # Validation data
        val_data = make_moons(n_samples=n, shuffle=True, noise=noise, random_state=rng_state)
        self.val_n = n
        self.val_xs = val_data[0]
        self.val_ys = self.reshape(val_data[1])

        # Test data
        test_data = make_moons(n_samples=n, shuffle=True, noise=noise, random_state=rng_state)
        self.test_n = n
        self.test_xs = test_data[0]
        self.test_ys = self.reshape(test_data[1])


class MNIST(DataSet):

    def __init__(self, data_path: str, key: JKey):
        data = np.load(data_path)

        self.n = 50000
        self.n_val = 10000
        self.n_test = 10000

        xs = jax.random.permutation(key, data['X'].reshape(60000, 784), axis=0)
        ys = jax.random.permutation(key, data['y'], axis=0)
        ys = jax.nn.one_hot(ys, 10)

        # Training data
        self.xs = xs[:self.n]
        self.ys = ys[:self.n]

        # Validation data
        self.val_xs = xs[self.n:]
        self.val_ys = ys[self.n:]

        # Test data
        self.test_xs = data['X_test'].reshape(self.n_test, 784)
        self.test_ys = jax.nn.one_hot(data['y_test'], 10)
