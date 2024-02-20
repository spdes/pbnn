import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from pbnn.solvers.resampling import systematic, stratified, sinkhorn, det_resampling
from functools import partial

jax.config.update("jax_enable_x64", True)
np.random.seed(666)


class TestClassicalResampling:

    @pytest.mark.parametrize('resampling_method', [systematic, stratified])
    def test_resamplings(self, resampling_method):
        s = 1000
        samples = jnp.asarray(np.random.randn(s, 2))
        weights = jnp.ones((s,))
        weights = weights / jnp.sum(weights)

        key = jax.random.PRNGKey(666)
        key, _ = jax.random.split(key)
        resampled_samples = resampling_method(samples, weights, key)
        resampled_weights = jnp.ones((s,)) / s

        true_mean = jnp.sum(samples * weights[:, None], axis=0)
        approx_mean = jnp.sum(resampled_samples * resampled_weights[:, None], axis=0)
        npt.assert_allclose(approx_mean, true_mean)


class TestOT:

    def test_sinkhorn(self):
        x1s, x2s = np.random.randn(2, 100)
        w1s, w2s = np.abs(np.random.randn(2, 100))
        w1s = w1s / w1s.sum()
        w2s = w2s / w2s.sum()

        def c(x1, x2): return (x1[:, None] - x2[None, :]) ** 2

        u0, v0 = jnp.ones((2, 100))
        eps = 1e-1
        niters = 1000

        u, v, coupling = sinkhorn((w1s, x1s), (w2s, x2s), c, eps, (u0, v0), niters)
        K = jnp.exp(-c(x1s, x2s) / eps)

        npt.assert_allclose(np.sum(coupling), 1.)
        npt.assert_allclose(u * (K @ v) - w1s, 0., atol=1e-5)
        npt.assert_allclose(v * (K.T @ u) - w2s, 0., atol=1e-5)


class TestDET:
    def test_det_resampling(self):
        r"""p(x | y) \propto N(y | x; H x, xi) N(x; m, v)
        """
        d = 2
        m, v = jnp.array([0., 0.]), jnp.array([[1., 0.2],
                                               [0.2, 1.]])
        H = jnp.array([1., 0.1])
        xi = 0.1
        y = 1.

        true_posterior_mean = v @ H / (jnp.dot(H, v @ H) + xi) * (y - jnp.dot(H, m))

        @partial(jax.vmap, in_axes=[None, 0])
        def pdf_y_cond_x(_y, _x):
            return jax.scipy.stats.norm.pdf(_y, jnp.dot(H, _x), jnp.sqrt(xi))

        nsamples = 1000

        key = jax.random.PRNGKey(666)
        samples = m + jnp.einsum('ij,ni->nj', jnp.linalg.cholesky(v),
                                 jax.random.normal(key, (nsamples, d)))
        weights = jnp.ones((nsamples,)) / nsamples
        posterior_weights = weights * pdf_y_cond_x(y, samples)
        posterior_weights = posterior_weights / jnp.sum(posterior_weights)
        posterior_resamples = det_resampling(samples, posterior_weights, None, weights,
                                             (jnp.ones((nsamples,)), jnp.ones((nsamples,))),
                                             niters=100, eps=0.1)

        npt.assert_allclose(jnp.sum(posterior_weights[:, None] * samples, axis=0),
                            jnp.mean(posterior_resamples, axis=0),
                            rtol=1e-7)
        npt.assert_allclose(jnp.mean(posterior_resamples, axis=0), true_posterior_mean, rtol=1e-1)

    def test_det_resampling_nongaussian(self):
        d = 2
        nsamples = 1000

        key = jax.random.PRNGKey(666)
        samples = jax.random.normal(key, (nsamples, d))
        weights = jnp.ones((nsamples,)) / nsamples

        @partial(jax.vmap, in_axes=[None, 0])
        def pdf_y_cond_x(_y, _x):
            return jax.scipy.stats.bernoulli.pmf(_y, jnp.tanh(_x[0] * _x[1]) + 1.)

        y = 1
        posterior_weights = weights * pdf_y_cond_x(y, samples)
        posterior_weights = posterior_weights / jnp.sum(posterior_weights)
        posterior_resamples = det_resampling(samples, posterior_weights, None, weights,
                                             (jnp.ones((nsamples,)), jnp.ones((nsamples,))),
                                             niters=100, eps=0.1)

        npt.assert_allclose(jnp.sum(posterior_weights[:, None] * samples, axis=0),
                            jnp.mean(posterior_resamples, axis=0),
                            rtol=1e-7)
