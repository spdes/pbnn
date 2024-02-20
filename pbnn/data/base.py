import jax
import jax.numpy as jnp
import numpy as np
from pbnn.typings import Array, JArray
from abc import ABCMeta
from typing import List, Tuple


class DataSet(metaclass=ABCMeta):
    n: int
    xs: Array
    ys: Array
    rnd_inds: List

    val_n: int
    val_xs: Array
    val_ys: Array

    test_n: int
    test_xs: Array
    test_ys: Array

    @staticmethod
    def reshape(x):
        if x.ndim == 0:
            return jnp.reshape(x, (1, 1))
        elif x.ndim == 1:
            return jnp.reshape(x, (-1, 1))
        else:
            return x

    @staticmethod
    def standardise(array):
        return (array - np.mean(array, axis=0)) / np.std(array, axis=0)

    def draw_subset(self, key, batch_size: int):
        inds = jax.random.choice(key, jnp.arange(self.n), (batch_size,), replace=False)
        return self.reshape(self.xs[inds, :]), self.reshape(self.ys[inds, :])

    def draw_subset_boundary(self, key, batch_size: int, boundary: int):
        pass

    def init_enumeration(self, key, batch_size: int):
        """Randomly split the data into `n / batch_size` chunks. If the divisor is not an integer, then use // which
        truncates the training data.
        """
        n_chunks = self.n // batch_size
        self.rnd_inds = jnp.array_split(jax.random.choice(key,
                                                          jnp.arange(batch_size * n_chunks), (batch_size * n_chunks,),
                                                          replace=False),
                                        n_chunks)

    def enumerate_subset(self, i: int):
        inds = self.rnd_inds[i]
        return self.reshape(self.xs[inds, :]), self.reshape(self.ys[inds, :])

    def normalise(self):
        # TODO
        pass

    @staticmethod
    def inflate_nan(ys: Array, xs: JArray) -> Tuple[JArray, JArray]:
        """Progressively inflate data with nan values. This is used for keeping the function signature jittable
        for SMC. Suggested by Adrien C.

        Parameters
        ----------
        ys, xs : Array (n, ...)

        Returns
        -------
        JArray, JArray (n - 1, n - 1, ...)
            Inflated two arrays.
        """
        # (n, dy) -> (n - 1, n - 1, dy)
        n, dy = ys.shape
        _, dx = xs.shape
        inflated_ys = np.empty((n - 1, n - 1, dy)) * np.nan
        inflated_xs = np.empty((n - 1, n - 1, dx)) * np.nan

        for i in range(n - 1):
            inflated_ys[i, :(i + 1)] = ys[:(i + 1)]
            inflated_xs[i, :(i + 1)] = xs[:(i + 1)]
        return jnp.asarray(inflated_ys), jnp.asarray(inflated_xs)
