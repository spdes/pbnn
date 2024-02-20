# Copyright (C) 2022 Zheng Zhao
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Filters and smoothers based on sequential Monte Carlo, for instance, particle solvers and smoothers.
"""
import jax
import jax.numpy as jnp
from pbnn.solvers.resampling import continuous_resampling_1d
from pbnn.data import DataSet
from pbnn.typings import JArray, FloatScalar, JKey, JFloat
from typing import Callable, Tuple, Any, Optional


def smc_kernel(samples: JArray, weights: JArray,
               ys: JArray, xs: JArray,
               transition_sampler: Callable[[JArray, JArray, JKey, Optional], Tuple[JArray, JArray]],
               transition_args: Any,
               measurement_cond_pdf: Callable[[JArray, JArray, JArray, Optional], JArray],
               measurement_args: Any,
               key: JKey,
               resampling_method: Callable[[JArray, JArray, JKey, JArray], JArray],
               resampling_threshold: float = 1.) -> Tuple[JArray, JArray, JFloat]:
    r"""The step kernel of the SMC sampler for sequential learning.

    Parameters
    ----------
    samples : JArray (s, dz)
        Samples, where `s` and `dz` are the number of samples and state dimension, respectively.
    weights : JArray (s, )
        The weights.
    ys : JArray (n, dy)
        Batch measurement points.
    xs : JArray (n, dx)
        The covariates. Batch input points, where `n` is the batch size.
    transition_sampler : Callable [(s, dz), (s, ), key, ...] -> (s, dz)
        A function that makes samples according to the transition distribution. Arguments are samples, weights, and
        other parameters.
    transition_args : Any
        The arguments passed to `transition_sampler`.
    measurement_args : Any
        The arguments passed to `measurement_cond_pdf`.
    measurement_cond_pdf : Callable [(n, dy), (s, dz), (n, dx)] -> (s, )
        The measurement PDF `p(ys | state; xs) = \prod_i^n p(y_i | state; x_i)`.
    key : JKey
        The random key.
    resampling_method : Callable [(s, dz), (s, ), key, (s, )] -> (s, dz)
        The resample method.
    resampling_threshold : float, default=1.
        The threshold for effective sample size. If the effective sample size is smaller than this threshold, then do
        resampling. The default is resampling at every step.

    Returns
    -------
    JArray (s, dz), JArray (s, ), JFloat
        Samples, weights, and (partial) negative log-likelihood.
    """
    s = samples.shape[0]

    # Markov transition
    samples, weights = transition_sampler(samples, weights, key, transition_args)

    # Reweight and normalise
    posterior_weights = weights * measurement_cond_pdf(ys, samples, xs, measurement_args)
    nell_inc = -jnp.log(jnp.sum(posterior_weights))
    posterior_weights = posterior_weights / jnp.sum(posterior_weights)

    # Resampling
    if resampling_threshold == 1:
        key, _ = jax.random.split(key)
        samples = resampling_method(samples, posterior_weights, key, weights)
        posterior_weights = jnp.ones((s,)) / s
    elif 0 < resampling_threshold < 1:
        key, _ = jax.random.split(key)
        ess = 1 / jnp.sum(posterior_weights ** 2)
        samples, posterior_weights = jax.lax.cond(ess <= resampling_threshold * s,
                                                  lambda _: (
                                                      resampling_method(samples, posterior_weights, key, weights),
                                                      jnp.ones((s,)) / s),
                                                  lambda _: (samples, posterior_weights),
                                                  None)
    return samples, posterior_weights, nell_inc


def smc_kernel_log(samples: JArray, log_weights: JArray,
                   ys: JArray, xs: JArray,
                   transition_sampler: Callable[[JArray, JArray, JKey, Optional], Tuple[JArray, JArray]],
                   transition_args: Any,
                   measurement_cond_log_pdf: Callable[[JArray, JArray, JArray, Optional], JArray],
                   measurement_args: Any,
                   key: JKey,
                   resampling_method: Callable[[JArray, JArray, JKey, Any], JArray],
                   resampling_threshold: float = 1.) -> Tuple[JArray, JArray, JFloat]:
    r"""See the docstring of `smc_kernel`, but the weights are now in the log domain for better numerical stability.
    """
    s = samples.shape[0]

    # Markov transition
    samples, log_weights = transition_sampler(samples, log_weights, key, transition_args)

    # Reweight and normalise
    log_posterior_weights = measurement_cond_log_pdf(ys, samples, xs, measurement_args) + log_weights
    _c = jax.scipy.special.logsumexp(log_posterior_weights)
    nell_inc = -_c
    log_posterior_weights = log_posterior_weights - _c

    # Resampling
    if resampling_threshold == 1.:
        key, _ = jax.random.split(key)
        samples = resampling_method(samples, jnp.exp(log_posterior_weights), key, jnp.exp(log_weights))
        log_posterior_weights = -jnp.log(s) * jnp.ones((s,))
    elif 0 < resampling_threshold < 1:
        key, _ = jax.random.split(key)
        log_ess = -jax.scipy.special.logsumexp(2 * log_posterior_weights)
        samples, log_posterior_weights = jax.lax.cond(log_ess <= jnp.log(resampling_threshold * s),
                                                      lambda _: (resampling_method(samples,
                                                                                   jnp.exp(log_posterior_weights), key,
                                                                                   jnp.exp(log_weights)),
                                                                 -jnp.log(s) * jnp.ones((s,))),
                                                      lambda _: (samples, log_posterior_weights),
                                                      None)
    return samples, log_posterior_weights, nell_inc


def chsmc(init_samples: JArray, init_weights: JArray,
          ys: JArray, xs: JArray,
          inflated_ys: JArray, inflated_xs: JArray,
          transition_sampler: Callable[[JArray, JArray, JKey, Optional], Tuple[JArray, JArray]],
          measurement_cond_pdf: Callable[[JArray, JArray, JArray, Optional], JArray],
          psi: JArray,
          key: JKey,
          log: bool,
          resampling_method: Callable[[JArray, JArray, JKey, JArray], JArray],
          resampling_threshold: float = 1.):
    if log:
        kernel = smc_kernel_log
    else:
        kernel = smc_kernel

    def identity_transition_sampler(*_):
        return init_samples, init_weights

    def scan_body(carry, elem):
        samples, weights, nell = carry
        y, x, inflated_y, inflated_x, _key = elem

        samples, weights, nell_inc = kernel(samples, weights,
                                            y[None, :], x[None, :],
                                            transition_sampler, (psi, inflated_y, inflated_x),
                                            measurement_cond_pdf, psi, _key, resampling_method,
                                            resampling_threshold)
        nell = nell + nell_inc
        return (samples, weights, nell), (samples, weights)

    # First iteration uses identity transition
    n = ys.shape[0]
    key, subkey = jax.random.split(key)
    f_samples, f_weights, f_nell = kernel(init_samples, init_weights,
                                          ys[0, None], xs[0, None],
                                          identity_transition_sampler, None,
                                          measurement_cond_pdf, psi, subkey, resampling_method,
                                          resampling_threshold)
    if n == 1:
        return f_samples, f_weights, f_nell
    else:
        keys = jax.random.split(key, num=n - 1)
        (f_samples, f_weights, f_nell), _ = jax.lax.scan(scan_body,
                                                         (f_samples, f_weights, f_nell),
                                                         (ys[1:], xs[1:], inflated_ys, inflated_xs, keys))
        return f_samples, f_weights, f_nell


def smc(init_samples: JArray, init_weights: JArray,
        ys: JArray, xs: JArray,
        transition_sampler: Callable[[JArray, JArray, JKey, Optional], Tuple[JArray, JArray]],
        transition_args: Any,
        measurement_cond_pdf: Callable[[JArray, JArray, JArray, Optional], JArray],
        measurement_args: Any,
        keys: JKey,
        log: bool,
        resampling_method: Callable[[JArray, JArray, JKey, JArray], JArray],
        resampling_threshold: float = 1.) -> Tuple[JArray, JArray, JFloat]:
    """Fixed-horizon sequential Monte Carlo sampler.

    Parameters
    ----------
    init_samples : JArray (s, dz)
        Samples from the initial random variable.
    init_weights : JArray (s, )
        Initial weights.
    ys
    xs
    transition_sampler
    transition_args
    measurement_cond_pdf
    measurement_args
    keys : JArray
        An array of random keys. `keys.shape[0]` must be equal to `ys.shape[0]`.
    log : bool
        Whether compute the weights in the log domain. If `True`, then everything related to weights must be in the log
        domain including the initial.
    resampling_method
    resampling_threshold

    Returns
    -------
    JArray (n, s, dz), JArray (n, s), JFloat
        Samples, weights, and negative log-likelihood.
    """
    if log:
        kernel = smc_kernel_log
    else:
        kernel = smc_kernel

    def scan_body(carry, elem):
        samples, weights, nell = carry
        y, x, key = elem

        samples, weights, nell_inc = kernel(samples, weights, y[None, :], x, transition_sampler, transition_args,
                                            measurement_cond_pdf, measurement_args, key, resampling_method,
                                            resampling_threshold)
        nell = nell + nell_inc
        return (samples, weights, nell), (samples, weights)

    (*_, f_nell), (f_samples, f_weights) = jax.lax.scan(scan_body,
                                                        (init_samples, init_weights, 0.),
                                                        (ys, xs, keys))
    return f_samples, f_weights, f_nell


def smc_random_markov(init_samples: JArray, init_weights: JArray,
                      dataset: DataSet, batch_size: int,
                      transition_sampler: Callable[[JArray, JArray, JKey, Optional], Tuple[JArray, JArray]],
                      measurement_cond_pdf: Callable[[JArray, JArray, JArray, Optional], JArray],
                      psi: JArray,
                      key: JKey,
                      log: bool,
                      resampling_method: Callable[[JArray, JArray, JKey, JArray], JArray],
                      resampling_threshold: float = 1.,
                      unbiased_batching: bool = True):
    if log:
        kernel = smc_kernel_log
    else:
        kernel = smc_kernel

    def scan_body(carry, elem):
        samples, weights, nell = carry
        y, x, _key, i = elem

        if unbiased_batching:
            batch_ys, batch_xs = jax.lax.cond(i < batch_size,
                                              lambda _: (dataset.ys[:batch_size, :], dataset.xs[:batch_size, :]),
                                              lambda _: dataset.draw_subset_boundary(_key, batch_size, i),
                                              None)
        else:
            batch_ys, batch_xs = dataset.draw_subset(_key, batch_size)

        _, subkey = jax.random.split(_key)
        samples, weights, nell_inc = kernel(samples, weights,
                                            y[None, :], x,
                                            transition_sampler, (psi, batch_ys, batch_xs),
                                            measurement_cond_pdf, psi, subkey, resampling_method,
                                            resampling_threshold)
        nell = nell + nell_inc
        return (samples, weights, nell), (samples, weights)

    keys = jax.random.split(key, num=dataset.n)
    (*_, f_nell), (f_samples, f_weights) = jax.lax.scan(scan_body,
                                                        (init_samples, init_weights, 0.),
                                                        (dataset.ys, dataset.xs, keys, jnp.arange(dataset.n)))
    return f_samples, f_weights, f_nell
