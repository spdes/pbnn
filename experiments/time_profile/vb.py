"""
MNIST.

batch_xs and batch_ys: train data.
xs and ys: validation data.
test_xs and test_ys: test data.

Variational Bayes.
"""
import argparse
import math
import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
from pbnn.data.classification import MNIST
from pbnn.solvers import variational_bayes
from pbnn.nn import make_pbnn_likelihood
from pbnn.experiment_settings.nn_configs import mnist
from functools import partial

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='MNIST using variational Bayes.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of MC samples.')
parser.add_argument('--vbsamples', type=int, default=100, help='The number of MC samples for VB approximation.')
parser.add_argument('--lr', type=float, default=1e-2, help='The learning rate for the optimizer.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
args = parser.parse_args()

print('Time-profiling VB.')

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
log_cond_pdf_likelihood_vmap = jax.vmap(log_cond_pdf_likelihood, in_axes=[None, 0, None, None])


# Prior definition
@partial(jax.vmap, in_axes=[0])
def log_pdf_prior(phi):
    return jnp.sum(jax.scipy.stats.norm.logpdf(phi, 0., math.sqrt(1.)))


# Make mean-field approximate posterior
@partial(jax.vmap, in_axes=[0, None])
def log_pdf_approx_posterior(phi, _theta):
    m, log_v = _theta[:shape_phi], _theta[shape_phi:]
    return jnp.sum(jax.scipy.stats.norm.logpdf(phi, m, jnp.exp(0.5 * log_v)))


def approx_posterior_sampler(_theta, _key, nsamples):
    m, log_v = _theta[:shape_phi], _theta[shape_phi:]
    return m + jax.random.normal(_key, (nsamples, shape_phi)) * jnp.exp(0.5 * log_v)


# Make ELBO and loss function
elbo = variational_bayes(log_cond_pdf_likelihood_vmap, log_pdf_prior, log_pdf_approx_posterior,
                         lambda u, v: approx_posterior_sampler(u, v, nsamples=args.vbsamples),
                         data_size=data_size)


def loss_fn(_param, _key, _ys, _xs):
    _psi, _theta = _param[:shape_psi], _param[shape_psi:]
    return -elbo(_psi, _theta, _key, _ys, _xs)

# Optax
@jax.jit
def opt_step_kernel(_param, _opt_state, _key, _ys, _xs):
    _loss, grad = jax.value_and_grad(loss_fn)(_param, _key, _ys, _xs)
    updates, _opt_state = optimiser.update(grad, _opt_state, _param)
    _param = optax.apply_updates(_param, updates)
    return _param, _opt_state, _loss


# Optimisation setup
optimiser = optax.adam(learning_rate=args.lr)
param = jnp.concatenate([pbnn_psi[0],
                         pbnn_phi[0],
                         jnp.ones((shape_phi,))])
opt_state = optimiser.init(param)
opt_nlpd = jnp.inf

# Optimisation loop
ts = np.zeros((1000, ))

loss = 0.
data_key, _ = jax.random.split(data_key)
dataset.init_enumeration(data_key, batch_size)
for i in range(1000):
    print(f'Iteration {i} / 1000')
    xs, ys = dataset.enumerate_subset(i)
    key, subkey = jax.random.split(key)
    tic = time.time()
    param, opt_state, loss = opt_step_kernel(param, opt_state, subkey, ys, xs)
    toc = time.time()
    ts[i] = toc - tic

np.save('./results/mnist/time_vb', ts)
