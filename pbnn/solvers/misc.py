import jax
import math
import blackjax
import jax.numpy as jnp
import numpy as np
from pbnn.typings import JArray, JKey, JFloat
from typing import Callable, Tuple


def variational_bayes(log_cond_pdf_likelihood: Callable[[JArray, JArray, JArray, JArray], JArray],
                      log_pdf_prior: Callable[[JArray], JArray],
                      log_pdf_approx_posterior: Callable[[JArray, JArray], JArray],
                      approx_posterior_sampler: Callable[[JArray, JKey], JArray],
                      data_size: int) -> Callable[
    [JArray, JArray, JKey, JArray, JArray], JFloat]:
    r"""Variational Bayes with stochastic approximations.

    .. math::

        E(\theta, \psi) = \int \log (\frac{p(y_{1:N} \mid \phi; \psi) \, p(\phi)}{q(\phi; \theta)} )
        q(\phi; \theta) d \phi.

    Parameters
    ----------
    log_cond_pdf_likelihood : (n, dy), (s, dw), (n, dx), (dp, ) -> (s, )
        The log conditional PDF of the likelihood.
    log_pdf_prior : (s, dw) -> (s, )
        The log PDF of the prior.
    log_pdf_approx_posterior : (s, dw), (dt, ) -> (s, )
        The log PDF of the approximate posterior distribution.
    approx_posterior_sampler : (dt, ), JKey -> (s, dw)
        A function that generates quadrature points based on the approximate posterior distribution.
    data_size : int
        The total data size.

    Returns
    -------
    Callable
        The (approximate) evidence lower bound function.
    """

    def elbo(psi: JArray, theta: JArray, key: JKey, ys: JArray, xs: JArray) -> JFloat:
        phis = approx_posterior_sampler(theta, key)
        return jnp.mean((log_cond_pdf_likelihood(ys, phis, xs, psi) * data_size / ys.shape[0] + log_pdf_prior(phis)
                         - log_pdf_approx_posterior(phis, theta)))

    return elbo


def maximum_a_posteriori(log_cond_pdf_likelihood: Callable[[JArray, JArray, JArray, JArray], JArray],
                         log_pdf_prior: Callable[[JArray], JArray],
                         data_size: int) -> Callable[[JArray, JArray, JArray, JArray], JFloat]:
    """Maximum a posterior estimation for both the parameters and latent variables.

    Parameters
    ----------
    log_cond_pdf_likelihood : (n, dy), (dw, ), (n, dx), (dp, ) -> float
        The log conditional PDF of the likelihood.
    log_pdf_prior : (dw, ) -> float
        The log PDF of the prior.
    data_size : int
        The total data size.

    Returns
    -------
    Callable
        The log posterior function.

    See the docstring of `variational_bayes`.
    """

    def ell(phi: JArray, psi: JArray, ys: JArray, xs: JArray):
        return log_cond_pdf_likelihood(ys, phi, xs, psi) * data_size / ys.shape[0] + log_pdf_prior(phi)

    return ell


def maximum_likelihood(log_cond_pdf_likelihood: Callable[[JArray, JArray, JArray, JArray], JArray],
                       prior_sampler: Callable[[JKey], JArray],
                       data_size: int) -> Callable[[JArray, JKey, JArray, JArray], JFloat]:
    """Brute-force computing the (lower bound of the) marginal likelihood by Monte Carlo samples.

    Parameters
    ----------
    log_cond_pdf_likelihood : (n, dy), (s, dw), (n, dx), (dp, ) -> (s, )
        The log conditional PDF of the likelihood.
    prior_sampler : JKey -> (s, dw)
        A function that generates quadrature points based on the prior distribution.
    data_size : int
        The total data size.
    """

    def ell_lb(psi: JArray, key: JKey, ys: JArray, xs: JArray):
        phis = prior_sampler(key)
        return jnp.mean(log_cond_pdf_likelihood(ys, phis, xs, psi) * data_size / ys.shape[0])

    return ell_lb


def laplace():
    pass


def hmc(log_pdf, init_sample, dt, integration_steps, inv_mass, nsamples, key, verbose: int = 0):
    """Hamiltonian Monte Carlo.

    Parameters
    ----------
    log_pdf : Callable (dw, ) -> JFloat
        The log PDF function.
    init_sample : JArray (dw, )
        The initial position.
    dt : float
        The ODE integration step length.
    integration_steps : int
        The number of ODE integrations.
    inv_mass : JArray (dw, ) or (dw, dw)
        Inverse of the mass matrix.
    nsamples : int
        The number of samples. Do thinning/burn-in by yourself.
    key : JKey
        A JAX random key.
    verbose : int, default=0
        Whether print the sampling iterations. Default is 0 which means do not print.

    Returns
    -------
    JArray (nsamples, dw)
        The HMC samples.
    """
    samples = np.zeros((nsamples, init_sample.shape[0]))

    hmc_obj = blackjax.hmc(log_pdf, dt, inv_mass, integration_steps)
    sampler = jax.jit(hmc_obj.step)
    state = hmc_obj.init(init_sample)

    for i in range(nsamples):
        key, _ = jax.random.split(key)
        state, _ = sampler(key, state)
        samples[i] = state.position

        if verbose:
            print(f'Sample: {i}')
    return samples


def langevin(samples: JArray, dt: float, nsteps: int, score_fn: Callable, key, *args):
    def scan_body(carry, elem):
        _samples = carry
        _key = elem
        _samples = _samples + 0.5 * score_fn(_samples, *args) * dt + math.sqrt(dt) * jax.random.normal(_key,
                                                                                                       _samples.shape)
        return _samples, None

    keys = jax.random.split(key, num=nsteps)
    samples, _ = jax.lax.scan(scan_body, samples, keys)
    return samples
