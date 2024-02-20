"""
Run synthetic regression experiments.

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
from pbnn.data.regression import OneDimGaussian
from pbnn.solvers import variational_bayes
from pbnn.nn import make_pbnn_likelihood
from pbnn.utils import nlpd_mc
from pbnn.experiment_settings.nn_configs import syn_regression
from functools import partial

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='Synthetic regression using variational Bayes.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of MC samples.')
parser.add_argument('--vbsamples', type=int, default=100, help='The number of MC samples for VB approximation.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
parser.add_argument('--data_size', type=int, default=100, help='Data size.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=200, help='The maximum number of iterations.')
args = parser.parse_args()

print('Running regression with VB.')

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


@jax.jit
def loss_fn(_param, _key, _ys, _xs):
    _psi, _theta = _param[:shape_psi], _param[shape_psi:]
    return -elbo(_psi, _theta, _key, _ys, _xs)


@jax.jit
def nlpd_fn(_samples, _psi, _ys, _xs):
    return nlpd_mc(lambda a, b, c, d: jnp.exp(log_cond_pdf_likelihood(a, b, c, d)),
                   _samples, _psi, _ys, _xs)


# Optax
@jax.jit
def opt_step_kernel(_param, _opt_state, _key, _ys, _xs):
    _loss, grad = jax.value_and_grad(loss_fn)(_param, _key, _ys, _xs)
    updates, _opt_state = optimiser.update(grad, _opt_state, _param)
    _param = optax.apply_updates(_param, updates)
    return _param, _opt_state, _loss


# Optimisation setup
optimiser = optax.adam(learning_rate=1e-2)
param = jnp.concatenate([pbnn_psi[0],
                         pbnn_phi[0],
                         jnp.ones((shape_phi,))])
opt_state = optimiser.init(param)
opt_nlpd = jnp.inf

# Optimisation loop
loss = 0.
for i in range(args.max_epochs):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        key, subkey = jax.random.split(key)
        param, opt_state, loss = opt_step_kernel(param, opt_state, subkey, ys, xs)

        # Save checkpoint
        psi, theta = param[:shape_psi], param[shape_psi:]
        samples = approx_posterior_sampler(theta, jax.random.split(subkey)[0], nsamples=args.nsamples)
        nlpd = nlpd_fn(samples, psi, dataset.val_ys, dataset.val_xs)
        if nlpd < opt_nlpd:
            test_nlpd = nlpd_fn(samples, psi, dataset.test_ys, dataset.test_xs)
            np.savez(f'./results/regression/vb_{args.id}',
                     i=i, j=j, samples=samples, psi=psi, nlpd=nlpd, test_nlpd=test_nlpd)
            opt_nlpd = nlpd
            print(f'Epoch: {i} / {args.max_epochs}, val_NLPD: {nlpd}, test_NLPD: {test_nlpd}')
        else:
            print(f'Epoch: {i} / {args.max_epochs}, val_NLPD: {nlpd}')
