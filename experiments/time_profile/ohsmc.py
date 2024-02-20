"""
MNIST.

batch_xs and batch_ys: train data.
val_xs and val_ys: validation data.
test_xs and test_ys: test data.

Open-horizon SMC.

Notes
-----
Set `nlpd_nchunks` be 10, and `batch_size` be 100 on Berzelius A100 80G. < 1hr / epoch w/o val.
"""
import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
from pbnn.data.classification import MNIST
from pbnn.solvers import smc_kernel_log, stratified
from pbnn.markov_kernels import make_random_walk
from pbnn.nn import make_pbnn_likelihood
from pbnn.experiment_settings.nn_configs import mnist

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='OH-SMC time profiling.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of SMC samples.')
parser.add_argument('--rw_var', type=float, default=1e-2, help='The random walk transition variance.')
parser.add_argument('--lr', type=float, default=1e-2, help='The learning rate for the optimizer.')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
args = parser.parse_args()

print('Time-profiling OHSMC.')

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

grad_log_cond_pdf_likelihood = jax.vmap(jax.grad(log_cond_pdf_likelihood, argnums=3), in_axes=[None, 0, None, None])
log_cond_pdf_likelihood_vmap = jax.vmap(log_cond_pdf_likelihood, in_axes=[None, 0, None, None])

# Prior definition
m0 = jnp.zeros((shape_phi,))
v0 = jnp.ones((shape_phi,))

# SMC setup
nsamples = args.nsamples
transition_sampler = make_random_walk(args.rw_var)


def resampling(_us, _ws, _key, _dummy):
    return stratified(_us, _ws, _key)


@jax.jit
def ohsmc(_samples, _log_weights, _psi, _opt_state, _key, _ys, _xs):
    _samples, _log_weights, _ = smc_kernel_log(_samples, _log_weights, _ys, _xs, transition_sampler, 1.,
                                               log_cond_pdf_likelihood_vmap, _psi, _key,
                                               resampling_method=stratified, resampling_threshold=1.)
    grad = -data_size / batch_size * jnp.einsum('i,ij->j',
                                                jnp.exp(_log_weights),
                                                grad_log_cond_pdf_likelihood(_ys, _samples, _xs, _psi))
    updates, _opt_state = optimiser.update(grad, _opt_state, _psi)
    _psi = optax.apply_updates(_psi, updates)
    return _samples, _log_weights, _psi, _opt_state

# Optimisation setup
optimiser = optax.adam(learning_rate=args.lr)
param = pbnn_psi[0]
opt_state = optimiser.init(param)
opt_nlpd = jnp.inf

# Optimisation loop
key, subkey = jax.random.split(key)
samples = m0 + jnp.sqrt(v0) * jax.random.normal(subkey, (nsamples, shape_phi))
log_weights = -jnp.log(nsamples) * jnp.ones((nsamples,))
psi = pbnn_psi[0]

ts = np.zeros((1000, ))

data_key, _ = jax.random.split(data_key)
dataset.init_enumeration(data_key, batch_size)
for i in range(1000):
    print(f'Iteration {i} / 1000')
    xs, ys = dataset.enumerate_subset(i)
    key, subkey = jax.random.split(key)
    tic = time.time()
    samples, log_weights, psi, opt_state = ohsmc(samples, log_weights, psi, opt_state, subkey, ys, xs)
    toc = time.time()
    ts[i] = toc - tic

np.save('./results/mnist/time_ohsmc', ts)
