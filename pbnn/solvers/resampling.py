"""
The codes in this file are selected and adapted from
https://github.com/AdrienCorenflos/parallel-ps/blob/e5a24fe0ba4afbf3275bcbfbd06908ace2ea7257/parallel_ps/core/resampling.py
which is now shipped as
a module in https://github.com/blackjax-devs/blackjax/blob/2bbdefc28cc7c2048431405fe5d47a1b76a69e64/blackjax/smc/resampling.py
under the Apache-2.0 license.

Here are the changes:

1. Simplified the signatures of systematic, stratified.
2. Removed the comments of multinomial.
3. Renamed rng_key to key.
4. Changed typing and the code formatting style.

The copyright notice:

Copyright 2020 The Blackjax developers, and Adrien Corenflos

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import jax
import jax.numpy as jnp
from pbnn.typings import JArray, JKey, FloatScalar
from functools import partial
from typing import Tuple, Callable, Any


def _sorted_uniforms(n, key: JKey) -> JArray:
    # Credit goes to Nicolas Chopin
    us = jax.random.uniform(key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]


def _systematic_or_stratified(samples: JArray, weights: JArray, key: JKey, is_systematic: bool) -> JArray:
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(key, ())
    else:
        u = jax.random.uniform(key, (n,))
    idx = jnp.searchsorted(jnp.cumsum(weights),
                           (jnp.arange(n, dtype=weights.dtype) + u) / n)
    return samples[jnp.clip(idx, 0, n - 1), ...]


def systematic(samples: JArray, weights: JArray, key: JKey, *args, **kwargs) -> JArray:
    return _systematic_or_stratified(samples, weights, key, True)


def stratified(samples: JArray, weights: JArray, key: JKey, *args, **kwargs) -> JArray:
    return _systematic_or_stratified(samples, weights, key, False)


def multinomial(samples: JArray, weights: JArray, key: JKey, *args, **kwargs) -> JArray:
    """Not tested.
    """
    n = weights.shape[0]
    idx = jnp.searchsorted(jnp.cumsum(weights),
                           _sorted_uniforms(n, key))
    return samples[jnp.clip(idx, 0, n - 1), ...]


def _avg_n_nplusone(x):
    hx = 0.5 * x
    y = jnp.pad(hx, [[0, 1]], constant_values=0.0, mode="constant")
    y = y.at[..., 1:].add(hx)
    return y


def continuous_resampling_1d(samples: JArray, weights: JArray, key: JKey, *args, **kwargs) -> JArray:
    """Continuous resampling for 1D state.

    This implementation is due to Adrien Corenflos.

    Parameters
    ----------
    samples : JArray (n, )
        Samples.
    weights : JArray (n, )
        Weights.
    key : JArray
        The jax random key.

    Returns
    -------
    JArray (n, )
        The resampled samples.

    References
    ----------
    Adrien Corenflos, James Thornton, George Deligiannidis, and Arnaud Doucet. Differentiable particle filtering via
    entropy-regularized optimal transport. In International Conference on Machine Learning 2021.

    Sheheryar Malik and Michael K. Pitt. Particle solvers for continuous likelihood evaluation and maximisation.
    Journal of Econometrics 2011.
    """
    nsamples = samples.shape[0]
    idx = jnp.argsort(samples)
    xs, ws = samples[idx], weights[idx]
    cs = jnp.cumsum(_avg_n_nplusone(ws))
    cs = cs[:-1]
    z = (jax.random.uniform(key, (nsamples,)) + jnp.arange(nsamples)) / nsamples
    return jnp.interp(z, cs, xs)


def sinkhorn(distribution1: Tuple[JArray, JArray],
             distribution2: Tuple[JArray, JArray],
             c: Callable[[JArray, JArray], JArray], eps: FloatScalar,
             init_potentials: Tuple[JArray, JArray],
             niters: int) -> Tuple[JArray, JArray, JArray]:
    """Sinkhorn algorithm for discrete-discrete entropy-regularised OT.

    Parameters
    ----------
    distribution1 : ((n, ), (n, d))
        A tuple of weights and samples.
    distribution2 : ((n, ), (n, d))
        A tuple of weights and samples.
    c : Callable (n, d), (m, d) -> (n, m)
        The cost function.
    eps : float-like
        The regularisation parameter.
    init_potentials : ((n, ), (n, ))
        The initial Kantorovich potentials.
    niters : int
        The number of iterations

    Returns
    -------
    JArray (n, ), JArray (n, ), JArray (n, n)
        Potentials u and v, and coupling P.
    """
    w1s, x1s = distribution1
    w2s, x2s = distribution2

    K = jnp.exp(-c(x1s, x2s) / eps)

    def scan_body(carry, _):
        _u, _v = carry

        _u = w1s / (K @ _v)
        _v = w2s / (K.T @ _u)
        return (_u, _v), None

    (u, v), _ = jax.lax.scan(scan_body, init_potentials, jnp.arange(niters))
    return u, v, K * u[:, None] * v


def det_resampling(samples: JArray, weights_v: JArray, _key: Any, weights_u: JArray,
                   u0_v0: Tuple[JArray, JArray], niters: int, eps: float = 0.1) -> JArray:
    """Differentiable resampling by solving an OT W(u, v) between two empirical measures.

    Parameters
    ----------
    samples : JArray (s, d)
        Samples.
    weights_u : JArray (s, )
        The target weights.
    _key : Any
        A placeholder for keeping consistence of the interface. Not used, you can pass any input.
    weights_v : JArray (s, )
        The original weights.
    u0_v0 : ((s, ), (s, ))
        Initial potentials.
    niters : int
        Number of iterations.
    eps : float
        Regularisation parameter.

    Returns
    -------
    JArray (s, d)
        The resampled samples.
    """

    @partial(jax.vmap, in_axes=[0, None])
    @partial(jax.vmap, in_axes=[None, 0])
    def c(x, y):
        return jnp.sum((x - y) ** 2)

    _, _, coupling = sinkhorn((weights_u, samples),
                              (weights_v, samples), c, eps, u0_v0, niters)
    return samples.shape[0] * coupling @ samples
