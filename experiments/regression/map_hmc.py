"""
Run synthetic regression experiments.

batch_xs and batch_ys: train data.
val_xs and val_ys: validation data.
test_xs and test_ys: test data.

Maximum a posteriori + HMC.

Note: Computing NLPD on the validation data means to run HMC at every iteration. This is not computationally possible
with the resources we have.
"""
import argparse
import math
import jax
import jax.numpy as jnp
import optax
import numpy as np
from pbnn.data.regression import OneDimGaussian
from pbnn.solvers import maximum_a_posteriori, hmc
from pbnn.nn import make_pbnn_likelihood
from pbnn.utils import nlpd_mc
from pbnn.experiment_settings.nn_configs import syn_regression

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='Synthetic regression using MAP-HMC.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of MCMC samples.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
parser.add_argument('--data_size', type=int, default=100, help='Data size.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=200, help='The maximum number of iterations.')
args = parser.parse_args()

print('Running regression with MAP-HMC.')

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


# Prior definition
def log_pdf_prior(phi):
    return jnp.sum(jax.scipy.stats.norm.logpdf(phi, 0., math.sqrt(1.)))


# Make ELBO and loss function
ell = maximum_a_posteriori(log_cond_pdf_likelihood, log_pdf_prior, data_size=data_size)


@jax.jit
def loss_fn(_param, _ys, _xs):
    phi, psi = _param[:shape_phi], _param[shape_phi:]
    return -ell(phi, psi, _ys, _xs)


@jax.jit
def nlpd_fn(_samples, _psi, _ys, _xs):
    return nlpd_mc(lambda a, b, c, d: jnp.exp(log_cond_pdf_likelihood(a, b, c, d)),
                   _samples, _psi, _ys, _xs)


# Optax
@jax.jit
def opt_step_kernel(_param, _opt_state, _ys, _xs):
    _loss, grad = jax.value_and_grad(loss_fn)(_param, _ys, _xs)
    updates, _opt_state = optimiser.update(grad, _opt_state, _param)
    _param = optax.apply_updates(_param, updates)
    return _param, _opt_state, _loss


# Optimisation setup
optimiser = optax.adam(learning_rate=1e-2)
param = jnp.concatenate([pbnn_phi[0], pbnn_psi[0]])
opt_state = optimiser.init(param)
opt_loss = jnp.inf
opt_param = None

# Optimisation loop
loss = 0.
for i in range(args.max_epochs):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        param, opt_state, loss = opt_step_kernel(param, opt_state, ys, xs)

        # Save checkpoint
        val_loss = loss_fn(param, dataset.val_ys, dataset.val_xs)
        if val_loss < opt_loss:
            opt_param = param
            opt_loss = val_loss
            print(f'Epoch: {i} / {args.max_epochs}, loss: {loss}, val_loss: {val_loss}')
        else:
            print(f'Epoch: {i} / {args.max_epochs}, loss: {loss}')

# Draw posterior samples based on the optimal param
opt_phi, opt_psi = opt_param[:shape_phi], opt_param[shape_phi:]


def log_posterior(phi):
    return log_cond_pdf_likelihood(dataset.ys, phi, dataset.xs, opt_psi) + log_pdf_prior(phi)


init_sample = opt_phi
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

np.savez(f'./results/regression/map_hmc_{args.id}',
         samples=samples, psi=opt_psi, nlpd=nlpd, test_nlpd=test_nlpd)
