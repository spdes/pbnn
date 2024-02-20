"""
MNIST.

batch_xs and batch_ys: train data.
xs and ys: validation data.
test_xs and test_ys: test data.

Maximum a posteriori + SWAG.

Implementation by Sebastian.

Note: See the note of `map_hmc.py`
"""
import argparse
import math
import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
from pbnn.data.classification import MNIST
from pbnn.solvers import maximum_a_posteriori
from pbnn.nn import make_pbnn_likelihood
from pbnn.experiment_settings.nn_configs import mnist

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='MNIST using MAP-SWAG.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of SWAG posterior samples.')
parser.add_argument('--swag_iters', type=int, default=50, help='The number of SWAG iterations.')
parser.add_argument('--k', type=int, default=100, help='The SWAG parameter K.')
parser.add_argument('--adam', action='store_true', help='Whether to use adam.')
parser.add_argument('--lr', type=float, default=1e-2, help='The learning rate for the optimizer.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
args = parser.parse_args()

print('Time-profiling SWAG.')

# Random key seed
# Separate the key branch for data and algorithm
key = jnp.asarray(np.load('./keys_jax.npy')[args.id])
data_key, key = jax.random.split(key)

# Dataset creation
dataset = MNIST('./data/mnist.npz', data_key)
data_size = dataset.n
batch_size = args.batch_size

# Define the pBNN
data_key, subkey = jax.random.split(data_key)
pbnn_phi, pbnn_psi, pbnn_forward_pass = mnist(subkey, batch_size)
shape_phi, shape_psi = pbnn_phi[0].shape[0], pbnn_psi[0].shape[0]

log_cond_pdf_likelihood, _, _ = make_pbnn_likelihood(pbnn_forward_pass, likelihood_type='categorical')


# Prior definition
def log_pdf_prior(phi):
    return jnp.sum(jax.scipy.stats.norm.logpdf(phi, 0., math.sqrt(1.)))


# Make ELBO and loss function
ell = maximum_a_posteriori(log_cond_pdf_likelihood, log_pdf_prior, data_size=data_size)


@jax.jit
def loss_fn(_param, _ys, _xs):
    _phi, _psi = _param[:shape_phi], _param[shape_phi:]
    return -ell(_phi, _psi, _ys, _xs)


# Optax
@jax.jit
def opt_step_kernel(_param, _opt_state, _ys, _xs):
    _loss, grad = jax.value_and_grad(loss_fn)(_param, _ys, _xs)
    updates, _opt_state = optimiser.update(grad, _opt_state, _param)
    _param = optax.apply_updates(_param, updates)
    return _param, _opt_state, _loss


# Optimisation setup
optimiser = optax.adam(learning_rate=args.lr) if args.adam else optax.sgd(learning_rate=args.lr)
param = jnp.concatenate([pbnn_phi[0], pbnn_psi[0]])
opt_state = optimiser.init(param)
opt_loss = jnp.inf
opt_param = None

# Optimisation loop 1: pre-train
ts = np.zeros((1000,))

loss = 0.
data_key, _ = jax.random.split(data_key)
dataset.init_enumeration(data_key, batch_size)
for i in range(1000):
    print(f'Iteration {i} / 1000')
    xs, ys = dataset.enumerate_subset(i)
    tic = time.time()
    param, opt_state, loss = opt_step_kernel(param, opt_state, ys, xs)
    toc = time.time()
    ts[i] = toc - tic

np.save('./results/mnist/time_swag', ts)
