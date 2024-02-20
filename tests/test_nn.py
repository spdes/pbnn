import jax.numpy as jnp
import jax
import pytest
import numpy.testing as npt
from pbnn.nn import make_pbnn, make_pbnn_likelihood, make_nn
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from functools import partial


@pytest.mark.parametrize('rnd_pos', [0, 1])
@pytest.mark.parametrize('likelihood_type', ['bernoulli', 'categorical', 0.11,
                                             jnp.array([[1., 0.2],
                                                        [0.2, 1.5]])])
def test_pbnn(rnd_pos, likelihood_type):
    if likelihood_type == 'bernoulli':
        out_dim = 1
        out_fn = nn.sigmoid
    elif likelihood_type == 'categorical':
        out_dim = 3
        out_fn = nn.softmax
    elif isinstance(likelihood_type, float):
        out_dim = 1
        out_fn = lambda u: u
    else:
        out_dim = 2
        out_fn = lambda u: u

    class MLP1(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=5, param_dtype=jax.numpy.float64)(x)
            x = nn.gelu(x)
            return x

    class MLP2(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=10, param_dtype=jax.numpy.float64)(x)
            x = nn.gelu(x)
            return x

    class MLP3(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=out_dim, param_dtype=jax.numpy.float64)(x)
            x = out_fn(x)
            return x

    class MLPTrue(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=5, param_dtype=jax.numpy.float64)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=10, param_dtype=jax.numpy.float64)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=out_dim, param_dtype=jax.numpy.float64)(x)
            x = out_fn(x)
            return x

    batch_size, in_dims = 20, [3, 5, 10]
    key = jax.random.PRNGKey(666)

    mlps = (MLP1(), MLP2(), MLP3(), MLPTrue())

    # Make pBNN
    keys = jax.random.split(key, 3)
    random_argnums = [rnd_pos, ]
    pbnn_rnd, pbnn_deterministic, pbnn_forward_pass, _ = make_pbnn(mlps[:-1],
                                                                   random_argnums=random_argnums,
                                                                   in_dims=in_dims,
                                                                   batch_size=batch_size,
                                                                   keys=keys)
    _, random_array_to_pytree = pbnn_rnd
    _, deterministic_array_to_pytree = pbnn_deterministic

    # Make the true NN
    key, _ = jax.random.split(keys[0])
    true_array_params, _, true_forward_pass, _ = make_nn(mlps[-1],
                                                         mlps[-1].init(key, jnp.ones((batch_size, in_dims[0]))))

    # Pseudo-params
    random_array = jnp.ones_like(pbnn_rnd[0])
    deterministic_array = jnp.ones_like(pbnn_deterministic[0]) * 2
    ls_mlp_array_to_dicts = [ravel_pytree(mlps[i].init(key, jnp.ones((batch_size, in_dims[i])))) for i in range(3)]
    ls_mlp_dicts_params = []
    for i in range(3):
        if i in random_argnums:
            ls_mlp_dicts_params.append(ls_mlp_array_to_dicts[i][1](jnp.ones(ls_mlp_array_to_dicts[i][0].shape)))
        else:
            ls_mlp_dicts_params.append(ls_mlp_array_to_dicts[i][1](2 * jnp.ones(ls_mlp_array_to_dicts[i][0].shape)))

    true_array, _ = ravel_pytree(ls_mlp_dicts_params)

    # Check size
    npt.assert_equal(random_array.shape[0] + deterministic_array.shape[0], true_array_params.shape[0])

    key, _ = jax.random.split(key)
    xss = jax.random.normal(key, (batch_size, in_dims[0]))

    # Check evaluation
    npt.assert_array_equal(pbnn_forward_pass(random_array, deterministic_array, xss),
                           true_forward_pass(true_array, xss))

    # Check likelihood
    # This checks the shape and execution only, not the results
    fns = make_pbnn_likelihood(pbnn_forward_pass, likelihood_type=likelihood_type)
    for fn in fns:
        _ = fn(jnp.ones((batch_size, out_dim)), random_array,
               jnp.ones((batch_size, in_dims[0])), deterministic_array)
