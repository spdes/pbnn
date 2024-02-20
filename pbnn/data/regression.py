import math
import jax.numpy as jnp
import jax.random
from pbnn.data import DataSet
from pbnn.typings import JArray, JKey
from typing import Callable


class OneDimGaussian(DataSet):
    n: int

    xi: float
    fs: Callable

    def __init__(self, key: JKey, n: int, xs: JArray = None, xi: float = 1.):
        self.n = n
        self.xi = xi

        # Training data
        if xs is None:
            xs = jnp.sort(jax.random.uniform(key, shape=(n, 1), minval=-6., maxval=6.), axis=0)
        xs = jnp.reshape(xs, (-1, 1))
        self.xs = xs
        self.fs = lambda u: u * jnp.sin(u * jnp.tanh(u))
        key, subkey = jax.random.split(key)
        self.ys = self.fs(xs) + math.sqrt(xi) * jax.random.normal(subkey, (n, 1))

        # Validation data
        key, subkey = jax.random.split(key)
        self.val_xs = jnp.sort(jax.random.uniform(subkey, shape=(n, 1), minval=-6., maxval=6.), axis=0)
        key, subkey = jax.random.split(key)
        self.val_ys = self.fs(self.val_xs) + math.sqrt(xi) * jax.random.normal(subkey, (n, 1))

        # Test data
        key, subkey = jax.random.split(key)
        self.test_xs = jnp.sort(jax.random.uniform(subkey, shape=(n, 1), minval=-6., maxval=6.), axis=0)
        key, subkey = jax.random.split(key)
        self.test_ys = self.fs(self.test_xs) + math.sqrt(xi) * jax.random.normal(subkey, (n, 1))
