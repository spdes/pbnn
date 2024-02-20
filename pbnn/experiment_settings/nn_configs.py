import jax
from flax import linen as nn

from pbnn.nn import make_pbnn

kernel_init = nn.initializers.xavier_normal()


def syn_regression(key, batch_size):
    """Model used for the synthetic regression experiments.
    """

    class NNBlock1(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=20, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            x = nn.gelu(x)
            return x

    class NNBlock2(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=10, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            x = nn.gelu(x)
            return x

    class NNBlock3(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=1, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            return x

    input_dims = [1, 20, 10]
    nns = (NNBlock1(), NNBlock2(), NNBlock3())
    random_argnums = (1,)
    keys = jax.random.split(key, num=len(nns))

    pbnn_phi, pbnn_psi, pbnn_forward_pass, _ = make_pbnn(nns, random_argnums, input_dims, batch_size, keys)
    return pbnn_phi, pbnn_psi, pbnn_forward_pass


def moon(key, batch_size):
    class NNBlock1(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=100, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=20, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            x = nn.gelu(x)
            return x

    class NNBlock2(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=5, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            x = nn.gelu(x)
            return x

    class NNBlock3(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=1, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            return x

    input_dims = [2, 20, 5]
    nns = (NNBlock1(), NNBlock2(), NNBlock3())
    random_argnums = (1,)
    keys = jax.random.split(key, num=len(nns))

    pbnn_phi, pbnn_psi, pbnn_forward_pass, _ = make_pbnn(nns, random_argnums, input_dims, batch_size, keys)
    return pbnn_phi, pbnn_psi, pbnn_forward_pass


def uci(key, batch_size, input_dim, output_dim):
    class NNBlock1(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=50, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=20, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            x = nn.gelu(x)
            return x

    class NNBlock2(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=5, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            x = nn.gelu(x)
            return x

    class NNBlock3(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=output_dim, param_dtype=jax.numpy.float64, kernel_init=kernel_init)(x)
            return x

    input_dims = [input_dim, 20, 5]
    nns = (NNBlock1(), NNBlock2(), NNBlock3())
    random_argnums = (1,)
    keys = jax.random.split(key, num=len(nns))

    pbnn_phi, pbnn_psi, pbnn_forward_pass, _ = make_pbnn(nns, random_argnums, input_dims, batch_size, keys)
    return pbnn_phi, pbnn_psi, pbnn_forward_pass


def mnist(key, batch_size):
    class CNNBlock1(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = x.reshape((x.shape[0], 28, 28, 1))
            x = nn.Conv(features=32, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            return x

    class CNNBlock2(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=64, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(features=256)(x)
            x = nn.relu(x)
            x = nn.Dense(features=10)(x)
            return x

    input_dims = [784, (14, 14, 32)]
    nns = (CNNBlock1(), CNNBlock2())
    random_argnums = (0,)
    keys = jax.random.split(key, num=len(nns))

    pbnn_phi, pbnn_psi, pbnn_forward_pass, _ = make_pbnn(nns, random_argnums, input_dims, batch_size, keys)
    return pbnn_phi, pbnn_psi, pbnn_forward_pass
