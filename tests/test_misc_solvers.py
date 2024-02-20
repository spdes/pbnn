import jax.numpy as jnp
import jax
import math
import pytest
import jaxopt
import numpy.testing as npt
from pbnn.solvers import variational_bayes, maximum_a_posteriori, hmc
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)

xs = 2.
psi = 0.12
likelihood_variance = 0.1
prior_mean = 1.1
prior_variance = 1.1
prior_nsamples = 10000
y_nsamples = 100

key = jax.random.PRNGKey(666)
_phi = prior_mean + math.sqrt(prior_variance) * jax.random.normal(key)
key, _ = jax.random.split(key)
ys = _phi + math.sqrt(likelihood_variance) * jax.random.normal(key, (y_nsamples,))

G = xs ** 2 * prior_variance * jnp.ones((y_nsamples, y_nsamples)) + likelihood_variance * jnp.eye(y_nsamples)
chol = jax.scipy.linalg.cho_factor(G)
c = xs * prior_variance * jnp.ones((y_nsamples,))
true_posterior_mean = prior_mean + jnp.dot(c, jax.scipy.linalg.cho_solve(chol, ys - (xs * prior_mean + psi)))
true_posterior_variance = prior_variance - jnp.dot(c, jax.scipy.linalg.cho_solve(chol, c))
true_marginal_log_likelihood = jax.scipy.stats.multivariate_normal.logpdf(ys,
                                                                          (xs * prior_mean + psi) * jnp.ones(
                                                                              (y_nsamples,)),
                                                                          G)


def test_vb_posterior():
    """When the approximate posterior distribution is the true distribution, the ELBO is exactly the marginal log
    likelihood.
    """

    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def log_cond_pdf_likelihood(_ys, phi, _xs, _psi):
        return jnp.sum(jax.scipy.stats.norm.logpdf(_ys, _xs * phi + _psi, math.sqrt(likelihood_variance)))

    @partial(jax.vmap, in_axes=[0])
    def log_pdf_prior(phi):
        return jnp.squeeze(jax.scipy.stats.norm.logpdf(phi, prior_mean, math.sqrt(prior_variance)))

    @partial(jax.vmap, in_axes=[0, None])
    def log_pdf_approx_posterior(phi, _theta):
        return jnp.squeeze(jax.scipy.stats.norm.logpdf(phi, _theta[0], jnp.exp(0.5 * _theta[1])))

    def approx_posterior_sampler(_theta, _key):
        return _theta[0] + jnp.exp(0.5 * _theta[1]) * jax.random.normal(_key, (prior_nsamples, 1))

    elbo = variational_bayes(log_cond_pdf_likelihood, log_pdf_prior, log_pdf_approx_posterior,
                             approx_posterior_sampler, data_size=y_nsamples)

    key = jax.random.PRNGKey(666)
    key, _ = jax.random.split(key)
    theta = jnp.array([true_posterior_mean, jnp.log(true_posterior_variance)])
    elbo_eval = elbo(psi, theta, key, ys, xs)
    npt.assert_allclose(elbo_eval, true_marginal_log_likelihood, rtol=1e-10)

    # Test when fixing psi
    def loss_fn(_theta):
        return -elbo(psi, _theta, key, ys, xs)

    solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=loss_fn)
    init_params = jnp.array([1., 1.])
    opt_params, opt_state = solver.run(init_params)
    npt.assert_allclose(opt_params[0], true_posterior_mean, rtol=1e-2)
    npt.assert_allclose(jnp.exp(opt_params[1]), true_posterior_variance, rtol=6e-2)

    # Test when fixing theta
    def loss_fn(_psi):
        return -elbo(_psi, jnp.array([true_posterior_mean, jnp.log(true_posterior_variance)]),
                     key, ys, xs)

    solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=loss_fn)
    init_params = jnp.array(1.)
    opt_params, opt_state = solver.run(init_params)
    npt.assert_allclose(opt_params, psi, rtol=1e-2)
    npt.assert_allclose(opt_state.fun_val, -true_marginal_log_likelihood, rtol=1e-2)

    # Test when jointly
    def loss_fn(_param):
        _psi, _theta = _param[0], _param[1:]
        return -elbo(_psi, _theta, key, ys, xs)

    solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=loss_fn)
    init_params = jnp.array([0., 0., 0.])
    opt_params, opt_state = solver.run(init_params)
    # The means estimates are not identifiable from each other
    # npt.assert_allclose(opt_params[0], psi, rtol=1e-2)
    # npt.assert_allclose(opt_params[1], true_posterior_mean, rtol=6e-2)
    npt.assert_allclose(jnp.exp(opt_params[2]), true_posterior_variance, rtol=1e-2)
    npt.assert_allclose(opt_state.fun_val, -true_marginal_log_likelihood, rtol=2e-2)


def test_map():
    def log_cond_pdf_likelihood(_ys, phi, _xs, _psi):
        return jnp.sum(jax.scipy.stats.norm.logpdf(_ys, _xs * phi + _psi, math.sqrt(likelihood_variance)))

    def log_pdf_prior(phi):
        return jnp.squeeze(jax.scipy.stats.norm.logpdf(phi, prior_mean, math.sqrt(prior_variance)))

    ell = maximum_a_posteriori(log_cond_pdf_likelihood, log_pdf_prior, data_size=y_nsamples)

    # When fixing psi, the ell must be stationary at the true phi.
    grad = jax.grad(ell, argnums=0)(true_posterior_mean, psi, ys, xs)
    npt.assert_allclose(grad, 0., atol=1e-10)


def test_mcmc():
    def log_pdf(x):
        return jnp.squeeze(jax.scipy.stats.norm.logpdf(x, 0., 1.))

    init_sample = jnp.array([0.])
    dt = 1e-2
    integration_steps = 100
    inv_mass = jnp.array([1.])

    nsamples = 2000
    key = jax.random.PRNGKey(666)
    samples = hmc(log_pdf, init_sample, dt, integration_steps, inv_mass, nsamples, key)

    npt.assert_allclose(jnp.mean(samples), 0., atol=1e-2)
    npt.assert_allclose(jnp.var(samples), 1., atol=1e-1)
