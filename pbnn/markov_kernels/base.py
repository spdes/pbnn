import jax
import math
import blackjax
import jax.numpy as jnp
from pbnn.typings import JArray, JKey
from functools import partial
from typing import Callable, Tuple


def make_simple_langevin_sampler(score: Callable):
    def transition_sampler_euler(samples, dt, key, *args):
        return samples + 0.5 * score(samples, *args) * dt + jnp.sqrt(dt) * jax.random.normal(key, samples.shape)

    return transition_sampler_euler


def make_pbnn_langevin(dt: float, nsteps: int, score_fn: Callable):
    def transition_sampler_euler(samples: JArray, weights: JArray, key: JKey, *args) -> Tuple[JArray, JArray]:
        def scan_body(carry, elem):
            _samples = carry
            _key = elem
            return (_samples + 0.5 * score_fn(_samples, *args) * dt
                    + math.sqrt(dt) * jax.random.normal(_key, _samples.shape), None)

        keys = jax.random.split(key, num=nsteps)
        samples, _ = jax.lax.scan(scan_body, samples, keys)
        return samples, weights

    return transition_sampler_euler


def make_pbnn_rwmrh(log_pdf, sigma, nsteps, homebrew: bool = True):
    """Random-walk Metropolis--Rosenbluth--Hastings.

    Parameters
    ----------
    log_pdf
    sigma
    nsteps
    homebrew : bool
        Set False to use Blackjax backend.

    Returns
    -------

    """

    def transition_sampler_blackjax(samples: JArray, weights: JArray, key: JKey, *args) -> Tuple[JArray, JArray]:
        def _log_pdf(u):
            return log_pdf(u, *args)

        def scan_body(carry, elem):
            _state = carry
            _key = elem
            _state, _ = rw.step(_key, _state)
            return _state, None

        @partial(jax.vmap, in_axes=[0, 0])
        def chain(init_state, chain_key):
            state = rw.init(init_state)
            step_keys = jax.random.split(chain_key, num=nsteps)
            state, _ = jax.lax.scan(scan_body, state, step_keys)
            return state.position

        rw = blackjax.additive_step_random_walk(_log_pdf, blackjax.mcmc.random_walk.normal(sigma))
        chain_keys = jax.random.split(key, samples.shape[0])
        samples = chain(samples, chain_keys)
        return samples, weights

    def transition_sampler(samples: JArray, weights: JArray, key: JKey, *args) -> Tuple[JArray, JArray]:
        @partial(jax.vmap, in_axes=[0, 0])
        def _rw_mrth(init_state, _key):
            return rw_mrth(log_pdf, init_state, sigma, nsteps, _key, *args)

        keys = jax.random.split(key, samples.shape[0])
        samples = _rw_mrth(samples, keys)
        return samples, weights

    if homebrew:
        return transition_sampler
    else:
        return transition_sampler_blackjax


def make_pbnn_mala(log_pdf, dt, nsteps):
    kernel = blackjax.mala(log_pdf, dt)

    def transition_sampler(samples: JArray, weights: JArray, key: JKey, *args) -> Tuple[JArray, JArray]:
        def scan_body(carry, elem):
            _state = carry
            _key = elem
            _state, _ = kernel.step(_key, _state)
            return state, None

        state = kernel.init(samples)
        keys = jax.random.split(key, num=nsteps)
        state, _ = jax.lax.scan(scan_body, state, keys)
        return state.position, weights

    return transition_sampler


def make_independent_ous(mean, variances, dt: float):
    z = jnp.exp(-0.5 * dt / variances)

    def transition_sampler(samples: JArray, weights: JArray, key: JKey, *args) -> Tuple[JArray, JArray]:
        return (
            samples * z + mean * (1 - z) + jnp.sqrt(variances * (1 - z ** 2)) * jax.random.normal(key, samples.shape),
            weights)

    return transition_sampler


def make_random_walk(variance: float):
    def transition_sampler(samples: JArray, weights: JArray, key: JKey, *args, r: float = 1.) -> Tuple[JArray, JArray]:
        return samples + jax.random.normal(key, samples.shape) * jnp.sqrt(variance * r), weights

    return transition_sampler


def make_adaptive_random_walk(lam: float, log: bool, whitened_cov: bool):
    """See Chopin 2022, pp. 332.

    Parameters
    ----------
    lam : float
        2.38 * d^{-0.5}.
    log : bool
        Whether the weights are in the log domain.
    whitened_cov : bool
        Whether whiten the proposal covariance.

    Returns
    -------

    Notes
    -----
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance.
    """

    def transition_sampler(samples: JArray, weights: JArray, key: JKey, *args) -> Tuple[JArray, JArray]:
        ws = jnp.exp(weights) if log else weights
        empirical_mean = jnp.sum(samples * ws[:, None], axis=0)
        res = samples - empirical_mean
        if whitened_cov:
            empirical_cov_diag = jnp.einsum('i,ij,ij->j', ws, res, res) / (1 - jnp.sum(ws ** 2)) * lam
            return samples + jax.random.normal(key, samples.shape) * jnp.sqrt(empirical_cov_diag), weights
        else:
            empirical_cov = jnp.einsum('i,ij,ik->jk', ws, res, res) / (1 - jnp.sum(ws ** 2)) * lam
            return samples + jax.random.normal(key, samples.shape) @ jnp.linalg.cholesky(empirical_cov), weights

    return transition_sampler


def rw_mrth(log_pdf, init_state, variance, steps, key, *args):
    def scan_body(carry, elem):
        state = carry
        _key = elem

        proposed_state = state + jnp.sqrt(variance) * jax.random.normal(_key, state.shape)
        log_acc_prob = jnp.minimum(0, log_pdf(proposed_state, *args) - log_pdf(state, *args))

        _, subkey = jax.random.split(_key)
        log_u = jnp.log(jax.random.uniform(subkey, minval=0, maxval=1))

        state = jax.lax.cond(log_u <= log_acc_prob,
                             lambda _: proposed_state,
                             lambda _: state,
                             None)
        return state, None

    keys = jax.random.split(key, steps)
    last_state, _ = jax.lax.scan(scan_body, init_state, keys)
    return last_state
