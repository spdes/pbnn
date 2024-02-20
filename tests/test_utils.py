import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from pbnn.utils import GaussianSum1D, GaussianSumND, nlpd_mc, nlpd_mc_seq, accuracy
from jax.config import config

config.update("jax_enable_x64", True)
np.random.seed(666)


class TestUtils:

    def test_gaussian_sum_1d(self):
        means = jnp.array([-1.1, 1.2])
        variances = jnp.array([0.1, 0.1])
        weights = jnp.array([0.4, 0.6])

        gs = GaussianSum1D.new(means, variances, weights)

        # Test pdf
        xs = jnp.linspace(-5, 5, 1000)
        npt.assert_allclose(jnp.trapz(gs.pdf(xs) * xs, xs), jnp.dot(weights, means))

        # Test sampling
        key = jax.random.PRNGKey(666)
        samples = gs.sampler(key, 100000)
        npt.assert_allclose(jnp.mean(samples), jnp.dot(weights, means), atol=1e-2, rtol=1e-2)

    def test_gaussian_sum_nd(self):
        d = 2

        means = jnp.array([[2., 2.],
                           [-1., -1.],
                           [-3, 3]])
        covs = jnp.array([[[0.2, 0.1],
                           [0.1, 1.]],
                          [[2., 0.2],
                           [0.2, 0.3]],
                          [[0.5, 0.],
                           [0., 1.]]])
        weights = jnp.array([0.4, 0.4, 0.2])

        gs = GaussianSumND.new(means, covs, weights)

        # Test logpdf
        x = jnp.asarray(np.random.randn(d))
        npt.assert_allclose(gs.logpdf(x), np.log(gs.pdf(x)), atol=1e-12, rtol=1e-12)

    def test_nlpd(self):
        def cond_pdf(y, phi, x, param):
            return jax.scipy.stats.norm.pdf(y, phi, 1.)

        key = jax.random.PRNGKey(666)
        samples = jax.random.normal(key, (10000,))

        ys = jnp.ones((100, 1))
        xs = jnp.ones((100, 1))

        computed_nlpd = nlpd_mc(cond_pdf, samples, None, ys, xs)
        true_nlpd = -jnp.mean(jax.scipy.stats.norm.logpdf(ys, 0., jnp.sqrt(2.)))
        npt.assert_allclose(computed_nlpd, true_nlpd, rtol=1e-2)

        seq_nlpd = nlpd_mc_seq(cond_pdf, samples, None, ys, xs, 5)
        npt.assert_allclose(seq_nlpd, computed_nlpd)

    def test_accuracy(self):
        predicted_logits = jnp.array([[0.1, 0.2, 0.7],
                                      [0.5, 0.4, 0.1],
                                      [0.0, 0.9, 0.1],
                                      [0.8, 0.1, 0.1],
                                      [0.7, 0.2, 0.1]])
        true_labels = jnp.array([[0, 0, 1],
                                 [1, 0, 0],
                                 [0, 0, 1],
                                 [1, 0, 0],
                                 [1, 0, 0]])
        true_acc = 4 / 5
        computed_acc = accuracy(predicted_logits, true_labels)
        npt.assert_array_equal(computed_acc, true_acc)
