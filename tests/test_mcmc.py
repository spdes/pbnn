import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from pbnn.markov_kernels import make_simple_langevin_sampler
from pbnn.utils import GaussianSumND
import matplotlib.pyplot as plt

np.random.seed(666)


class TestSGLD:
    """Test stochastic gradient Langevin dymanics.
    """

    def test_1d_normal(self):
        mean, scale = 5., 3.

        def score(x):
            return jax.grad(jax.scipy.stats.norm.logpdf, argnums=0)(x, mean, scale)

        n = 5000
        dt = scale
        key = jax.random.PRNGKey(666)
        sampler = jax.jit(make_simple_langevin_sampler(score))
        samples = np.zeros((n,))

        sample = jnp.array(0.)
        for i in range(n):
            key, _ = jax.random.split(key)
            sample = sampler(sample, dt, key)
            samples[i] = sample

        samples = samples[1000:]
        npt.assert_allclose(np.mean(samples), mean, rtol=1e-1)
        npt.assert_allclose(np.std(samples), scale, rtol=1e-1)
