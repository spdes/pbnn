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
from pbnn.data.classification import MNIST
from pbnn.solvers import smc_kernel_log, stratified
from pbnn.markov_kernels import make_random_walk
from pbnn.nn import make_pbnn_likelihood
from pbnn.utils import nlpd_mc_seq
from pbnn.experiment_settings.nn_configs import mnist

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='MNIST using OH-SMC.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of SMC samples.')
parser.add_argument('--rw_var', type=float, default=1e-2, help='The random walk transition variance.')
parser.add_argument('--lr', type=float, default=1e-2, help='The learning rate for the optimizer.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
parser.add_argument('--batch_size', type=int, default=100, help='The batch size.')
parser.add_argument('--nlpd_nchunks', type=int, default=10,
                    help='The number of splits for computing validation error.')
parser.add_argument('--nlpd_reduce', type=int, default=10,
                    help='Reduce the computation for NLPD by a factor.')
parser.add_argument('--max_epochs', type=int, default=10, help='The maximum number of epochs.')
args = parser.parse_args()

print('Running mnist with OHSMC.')

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


@jax.jit
def nlpd_fn(_samples, _psi, _ys, _xs):
    return nlpd_mc_seq(lambda a, b, c, d: jnp.exp(log_cond_pdf_likelihood(a, b, c, d)),
                       _samples, _psi, _ys, _xs, args.nlpd_nchunks)


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
for i in range(args.max_epochs):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(data_size // batch_size):
        xs, ys = dataset.enumerate_subset(j)
        key, subkey = jax.random.split(key)
        samples, log_weights, psi, opt_state = ohsmc(samples, log_weights, psi, opt_state, subkey, ys, xs)

        # Save checkpoint
        # This will take a while
        nlpd = nlpd_fn(samples[::args.nlpd_reduce], psi, dataset.val_ys, dataset.val_xs)
        if nlpd < opt_nlpd:
            test_nlpd = nlpd_fn(samples[::args.nlpd_reduce], psi, dataset.test_ys, dataset.test_xs)
            np.savez(f'./results/mnist/ohsmc_{args.id}',
                     i=i, samples=samples, log_weights=log_weights, psi=psi, nlpd=nlpd, test_nlpd=test_nlpd)
            opt_nlpd = nlpd
            print(f'Epoch: {i} / {args.max_epochs}, val_NLPD: {nlpd}, test_NLPD: {test_nlpd}')
        else:
            print(f'Epoch: {i} / {args.max_epochs}, val_NLPD: {nlpd}')
