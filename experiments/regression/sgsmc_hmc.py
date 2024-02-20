"""
Run synthetic regression experiments.

batch_xs and batch_ys: train data.
val_xs and val_ys: validation data.
test_xs and test_ys: test data.

Stochastic gradient SMC.
"""
import argparse
import math
import jax
import jax.numpy as jnp
import optax
import numpy as np
from pbnn.data.regression import OneDimGaussian
from pbnn.solvers import chsmc, stratified, hmc
from pbnn.markov_kernels import make_random_walk, make_adaptive_random_walk, make_pbnn_rwmrh
from pbnn.nn import make_pbnn_likelihood
from pbnn.utils import nlpd_mc
from pbnn.experiment_settings.nn_configs import syn_regression

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='Synthetic regression using SG-SMC.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of SMC samples.')
parser.add_argument('--rw_var', type=float, default=5e-2, help='The MRTH random walk transition variance.')
parser.add_argument('--rw_steps', type=int, default=10, help='The number of MRTH random walk steps.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
parser.add_argument('--data_size', type=int, default=100, help='Data size.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=200, help='The maximum number of iterations.')
args = parser.parse_args()

print('Running regression with SGSMC-HMC.')

# Random key seed
# Separate the key branch for data and algorithm
key = jnp.asarray(np.load('./keys_jax.npy')[args.id])
data_key, key = jax.random.split(key)

# Dataset creation
data_size = args.data_size
batch_size = args.batch_size
dataset = OneDimGaussian(key=data_key, n=data_size)

# Define the pBNN
data_key, subkey = jax.random.split(data_key)
pbnn_phi, pbnn_psi, pbnn_forward_pass = syn_regression(subkey, batch_size)
shape_phi, shape_psi = pbnn_phi[0].shape[0], pbnn_psi[0].shape[0]

log_cond_pdf_likelihood, _, _ = make_pbnn_likelihood(pbnn_forward_pass, likelihood_type=dataset.xi)
grad_log_cond_pdf_likelihood = jax.vmap(jax.grad(log_cond_pdf_likelihood, argnums=3), in_axes=[None, 0, None, None])
log_cond_pdf_likelihood_vmap = jax.vmap(log_cond_pdf_likelihood, in_axes=[None, 0, None, None])

# Prior definition
m0 = jnp.zeros((shape_phi,))
v0 = jnp.ones((shape_phi,))


def log_pdf_prior(phi):
    return jnp.sum(jax.scipy.stats.norm.logpdf(phi, 0., math.sqrt(1.)))


# SMC setup
nsamples = args.nsamples


def log_posterior(_phi, _args):
    _psi, _inflated_y, _inflated_x = _args
    return log_cond_pdf_likelihood(_inflated_y, _phi, _inflated_x, _psi) + log_pdf_prior(_phi)


transition_sampler = make_pbnn_rwmrh(log_posterior, args.rw_var, args.rw_steps)


def resampling(_us, _ws, _key, _dummy):
    return stratified(_us, _ws, _key)


@jax.jit
def sgsmc(_samples, _log_weights, _psi, _opt_state, _key, _ys, _xs, _inflated_ys, _inflated_xs):
    _samples = m0 + jnp.sqrt(v0) * jax.random.normal(_key, (nsamples, shape_phi))
    _log_weights = -jnp.log(nsamples) * jnp.ones((nsamples,))
    _, _subkey = jax.random.split(_key)
    _samples, _log_weights, _nell = chsmc(_samples, _log_weights, _ys, _xs, _inflated_ys, _inflated_xs,
                                          transition_sampler,
                                          log_cond_pdf_likelihood_vmap, _psi, _subkey, True,
                                          resampling_method=stratified, resampling_threshold=1.)
    grad = -data_size / batch_size * jnp.einsum('i,ij->j',
                                                jnp.exp(_log_weights),
                                                grad_log_cond_pdf_likelihood(_ys, _samples, _xs, _psi))
    updates, _opt_state = optimiser.update(grad, _opt_state, _psi)
    _psi = optax.apply_updates(_psi, updates)
    return _samples, _log_weights, _psi, _opt_state, _nell


@jax.jit
def nlpd_fn(_samples, _psi, _ys, _xs):
    return nlpd_mc(lambda a, b, c, d: jnp.exp(log_cond_pdf_likelihood(a, b, c, d)),
                   _samples, _psi, _ys, _xs)


# Optimisation setup
optimiser = optax.adam(learning_rate=1e-2)
param = pbnn_psi[0]
opt_state = optimiser.init(param)
opt_nlpd = jnp.inf

# Optimisation loop
key, subkey = jax.random.split(key)
samples = m0 + jnp.sqrt(v0) * jax.random.normal(subkey, (nsamples, shape_phi))
log_weights = -jnp.log(nsamples) * jnp.ones((nsamples,))
psi = pbnn_psi[0]
opt_psi = None
for i in range(args.max_epochs):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        inflated_xs, inflated_ys = dataset.inflate_nan(xs, ys)
        key, subkey = jax.random.split(key)
        samples, log_weights, psi, opt_state, nell = sgsmc(samples, log_weights, psi, opt_state, subkey,
                                                           ys, xs, inflated_ys, inflated_xs)

        # Save checkpoint
        nlpd = nlpd_fn(samples, psi, dataset.val_ys, dataset.val_xs)
        if nlpd < opt_nlpd:
            opt_psi = psi
            opt_nlpd = nlpd
            print(f'Epoch: {i} / {args.max_epochs}, val_NLPD: {nlpd}, *')
        else:
            print(f'Epoch: {i} / {args.max_epochs}, val_NLPD: {nlpd}')


# Draw posterior samples based on the optimal param
def log_posterior(phi):
    return log_cond_pdf_likelihood(dataset.ys, phi, dataset.xs, opt_psi) + log_pdf_prior(phi)


init_sample = samples[0]
dt = 1e-2
integration_steps = 100
inv_mass = jnp.ones_like(init_sample)

burn_in = 2000
nsamples = burn_in + args.nsamples
key, subkey = jax.random.split(key)
samples = hmc(log_posterior, init_sample, dt, integration_steps, inv_mass, nsamples, subkey, verbose=0)[burn_in:]

nlpd = nlpd_fn(samples, opt_psi, dataset.ys, dataset.xs)
test_nlpd = nlpd_fn(samples, opt_psi, dataset.test_ys, dataset.test_xs)
print(f'NLPD: {nlpd}, test_NLPD: {test_nlpd}')

np.savez(f'./results/regression/sgsmc_hmc_{args.id}',
         samples=samples, psi=opt_psi, nlpd=nlpd, test_nlpd=test_nlpd)
