import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from pbnn.solvers.smc import smc_kernel, smc_kernel_log, smc
from pbnn.solvers.resampling import systematic, multinomial, stratified, det_resampling
from pbnn.markov_kernels.base import make_pbnn_langevin
from functools import partial

jax.config.update("jax_enable_x64", True)


class TestSMCKernel:

    @pytest.mark.parametrize('resampling_method', [systematic, multinomial, stratified, det_resampling])
    def test_smc_kernels(self, resampling_method):
        np.random.seed(666)

        dx, s, n, dy = 2, 1000, 1, 2
        samples = jnp.asarray(np.random.randn(s, dx))
        weights = jnp.ones((s,)) / s
        log_weights = jnp.log(weights)

        xs = jnp.asarray(np.random.randn(n, dx))
        ys = jnp.asarray(np.random.randn(n, dy))
        H = jnp.asarray(np.random.randn(dy, dx))

        true_mean = jnp.eye(dx) @ H.T @ jnp.linalg.inv(H @ jnp.eye(dx) @ H.T + jnp.eye(dy)) @ (ys - xs)[0]

        def transition_sampler(_samples, _weights, _key, _):
            return _samples + 0. * jax.random.normal(_key, (s, dx)), _weights

        def pdf_y_cond(y, sample, x):
            return jax.scipy.stats.multivariate_normal.pdf(y, H @ sample + x, jnp.eye(dy))

        def log_pdf_y_cond(y, sample, x):
            return jax.scipy.stats.multivariate_normal.logpdf(y, H @ sample + x, jnp.eye(dy))

        def measurement_cond_pdf(_ys, _samples, _xs, _):
            val = jax.vmap(jax.vmap(pdf_y_cond, in_axes=[None, 0, None]), in_axes=[0, None, 0])(_ys, _samples, _xs)
            return jnp.prod(val, axis=0)

        def measurement_cond_log_pdf(_ys, _samples, _xs, _):
            val = jax.vmap(jax.vmap(log_pdf_y_cond, in_axes=[None, 0, None]), in_axes=[0, None, 0])(_ys, _samples, _xs)
            return jnp.sum(val, axis=0)

        key = jax.random.PRNGKey(666)
        key, _ = jax.random.split(key)
        det_init = (jnp.ones((s,)), jnp.ones((s,)))

        def resampling(_samples, _weights, _key, _wus):
            return resampling_method(_samples, _weights, _key, weights_u=_wus,
                                     u0_v0=det_init, niters=50, eps=1.)

        results = smc_kernel(samples, weights, ys, xs, transition_sampler, None, measurement_cond_pdf, None, key,
                             resampling, resampling_threshold=1.)
        results_log = smc_kernel_log(samples, log_weights, ys, xs, transition_sampler, None, measurement_cond_log_pdf,
                                     None, key, resampling, resampling_threshold=1.)

        npt.assert_allclose(results[0], results_log[0], rtol=1e-6)
        npt.assert_allclose(results[1], jnp.exp(results_log[1]), rtol=1e-6)
        npt.assert_allclose(results[2], results_log[2], rtol=1e-6)

        samples, weights = results_log[0], jnp.exp(results_log[1])
        npt.assert_allclose(jnp.sum(samples * weights[:, None], axis=0), true_mean, rtol=1e-1)


class TestSMC:

    @pytest.mark.parametrize('resampling_method', [systematic, multinomial, stratified, det_resampling])
    def test_smc_vs_smc_kernel(self, resampling_method):
        np.random.seed(666)
        nsamples = 50

        dx, dy = 2, 2

        init_samples = jnp.asarray(np.random.randn(nsamples, dx))
        init_weights = jnp.ones((nsamples,)) / nsamples

        xs = jnp.asarray(np.random.randn(1, dx))
        ys = jnp.asarray(np.random.randn(1, dy))

        def transition_sampler(_samples, _weights, _key, _):
            return _samples + jax.random.normal(_key, (nsamples, dx)), _weights

        def measurement_cond_pdf(_y, _samples, _xs, _):
            return jax.scipy.stats.multivariate_normal.pdf(_y[0], _samples, jnp.eye(dy))

        key = jax.random.PRNGKey(666)
        keys = jax.random.split(key, num=1)
        det_init = (jnp.ones((nsamples,)), jnp.ones((nsamples,)))

        def resampling(_samples, _weights, _key, _wus):
            return resampling_method(_samples, _weights, _key, weights_u=_wus,
                                     u0_v0=det_init, niters=50, eps=1.)

        step_results = smc_kernel(init_samples, init_weights, ys, xs,
                                  transition_sampler, None, measurement_cond_pdf, None, keys[0],
                                  resampling, resampling_threshold=1.)
        results = smc(init_samples, init_weights, ys, xs, transition_sampler, None, measurement_cond_pdf, None,
                      keys, log=False, resampling_method=resampling, resampling_threshold=1.)
        for (i, j) in zip(step_results, results):
            npt.assert_allclose(i, jnp.squeeze(j), rtol=1e-10)

    @pytest.mark.parametrize('log', [True, False])
    @pytest.mark.parametrize('resampling_method', [systematic, stratified, det_resampling])
    def test_vs_kf(self, log, resampling_method):
        F = jnp.array([[1., 0.],
                       [-0.1, 1.]])
        Q = jnp.eye(2) * 0.1
        H = jnp.ones((1, 2))
        xi = jnp.eye(1) * 0.1

        m0 = jnp.zeros((2,))
        v0 = jnp.eye(2)
        state0 = jnp.zeros((2,))

        n = 10
        xs = jnp.linspace(0., 1., n)

        def simulate_scan(carry, elem):
            state = carry
            x, _key = elem

            state = F @ state + jnp.linalg.cholesky(Q) @ jax.random.normal(_key, (2,))
            _key, _ = jax.random.split(_key)
            y = H @ state + x + jnp.linalg.cholesky(xi) @ jax.random.normal(_key, (1,))
            return state, (state, y)

        key = jax.random.PRNGKey(666)
        keys = jax.random.split(key, n)
        _, (_, ys) = jax.lax.scan(simulate_scan, state0, (xs, keys))

        def kf_scan(carry, elem):
            mf, vf, nell = carry
            y, x = elem

            mp, vp = F @ mf, F @ vf @ F.T + Q
            S = H @ vp @ H.T + xi
            K = vp @ H.T / S

            pred_y = H @ mp + x
            mf = mp + K @ (y - pred_y)
            vf = vp - K @ K.T * S
            nell = nell - jnp.squeeze(jax.scipy.stats.norm.logpdf(y, pred_y, jnp.sqrt(S)))
            return (mf, vf, nell), (mf, vf, nell)

        _, (mfs, vfs, nells) = jax.lax.scan(kf_scan, (m0, v0, 0.), (ys, xs))

        nsamples = 2000

        def transition_sampler(_samples, _weights, _key, *args):
            return (jnp.einsum('ij,nj->ni',
                               F, _samples) + jnp.einsum('ij,nj->ni',
                                                         jnp.linalg.cholesky(Q),
                                                         jax.random.normal(_key, (nsamples, 2))),
                    _weights)

        @partial(jax.vmap, in_axes=[None, 0, None, None])
        def measurement_cond_pdf(_y, _sample, _x, _):
            return jnp.squeeze(jax.scipy.stats.norm.pdf(_y, H @ _sample + _x, jnp.sqrt(xi)))

        @partial(jax.vmap, in_axes=[None, 0, None, None])
        def measurement_cond_log_pdf(_y, _sample, _x, _):
            return jnp.squeeze(jax.scipy.stats.norm.logpdf(_y, H @ _sample + _x, jnp.sqrt(xi)))

        init_samples = m0 + jnp.einsum('ij,nj->ni', jnp.linalg.cholesky(v0), jax.random.normal(key, (nsamples, 2)))
        init_weights = jnp.ones((nsamples,)) / nsamples
        if log:
            init_weights = jnp.log(init_weights)

        det_init = (jnp.ones((nsamples,)), jnp.ones((nsamples,)))

        def resampling(_samples, _weights, _key, _wus):
            return resampling_method(_samples, _weights, _key, weights_u=_wus,
                                     u0_v0=det_init, niters=50, eps=1.)

        keys = jax.random.split(keys[0], n)

        samples, weights, nell_approx = smc(init_samples, init_weights, ys, xs, transition_sampler, None,
                                            measurement_cond_log_pdf if log else measurement_cond_pdf,
                                            _,
                                            keys, log=log,
                                            resampling_method=resampling, resampling_threshold=1.)
        npt.assert_allclose(nell_approx, nells[-1], rtol=5e-2)
        npt.assert_allclose(jnp.mean(samples, 1), mfs, atol=7e-1)


def test_transition_sampler_langevin():
    @partial(jax.vmap, in_axes=[0])
    def score(x):
        return jax.grad(jax.scipy.stats.norm.logpdf)(x, 1., 2.)

    dt = 1e-1
    nsteps = 1000
    transition_sampler = make_pbnn_langevin(dt, nsteps, score)

    key = jax.random.PRNGKey(666)
    samples = jax.random.normal(key, (10000,))
    samples, _ = transition_sampler(samples, 0., key)

    npt.assert_allclose(jnp.mean(samples), 1., rtol=2e-2)
    npt.assert_allclose(jnp.var(samples), 4., rtol=2e-2)
