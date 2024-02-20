"""
Run experiments in the two moon datasets.

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
from pbnn.data.classification import Moons
from pbnn.solvers import maximum_a_posteriori
from pbnn.nn import make_pbnn_likelihood
from pbnn.utils import nlpd_mc
from pbnn.experiment_settings.nn_configs import moon

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='Synthetic classification using MAP-SWAG.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of SWAG posterior samples.')
parser.add_argument('--swag_iters', type=int, default=200, help='The number of SWAG iterations.')
parser.add_argument('--k', type=int, default=100, help='The SWAG parameter K.')
parser.add_argument('--adam', action='store_true', help='Whether to use adam.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
parser.add_argument('--data_size', type=int, default=100, help='Data size.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=100, help='The maximum number of iterations.')
args = parser.parse_args()

print('Running moon with SWAG-HMC.')

# Random key seed
# Separate the key branch for data and algorithm
key = jnp.asarray(np.load('./keys_jax.npy')[args.id])
data_key, key = jax.random.split(key)

# Dataset creation
data_size = args.data_size
batch_size = args.batch_size
dataset = Moons(n=data_size,
                rng_state=np.random.RandomState(np.load('./keys_np.npy')[args.id]),
                noise=0.3)

# Define the pBNN
data_key, subkey = jax.random.split(data_key)
pbnn_phi, pbnn_psi, pbnn_forward_pass = moon(subkey, batch_size)
shape_phi, shape_psi = pbnn_phi[0].shape[0], pbnn_psi[0].shape[0]

log_cond_pdf_likelihood, _, _ = make_pbnn_likelihood(pbnn_forward_pass, likelihood_type='bernoulli')


# Prior definition
def log_pdf_prior(phi):
    return jnp.sum(jax.scipy.stats.norm.logpdf(phi, 0., math.sqrt(1.)))


# Make ELBO and loss function
ell = maximum_a_posteriori(log_cond_pdf_likelihood, log_pdf_prior, data_size=data_size)


@jax.jit
def loss_fn(_param, _ys, _xs):
    _phi, _psi = _param[:shape_phi], _param[shape_phi:]
    return -ell(_phi, _psi, _ys, _xs)


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
optimiser = optax.adam(learning_rate=1e-2) if args.adam else optax.sgd(learning_rate=1e-2)
param = jnp.concatenate([pbnn_phi[0], pbnn_psi[0]])
opt_state = optimiser.init(param)
opt_loss = jnp.inf
opt_param = None

# Optimisation loop 1: pre-train
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

# Optimisation loop 2: sampling
opt_psi = opt_param[shape_phi:]
param = opt_param
K = args.k
phi_bar = opt_param[:shape_phi]
phi2_bar = phi_bar ** 2
D = []
for i in range(args.swag_iters):
    key, subkey = jax.random.split(key)
    dataset.init_enumeration(subkey, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        param, opt_state, loss = opt_step_kernel(param, opt_state, ys, xs)

    phi = param[:shape_phi]
    phi_bar = (i * phi_bar + phi) / (i + 1)
    phi2_bar = (i * phi2_bar + phi ** 2) / (i + 1)
    D.append(phi - phi_bar)
    D = D[-K:]
mu_swag = phi_bar
sigma_diag_swag = phi2_bar - phi_bar ** 2
D = jnp.array(D).T
sigma_low_rank_swag = 1. / (K + 1) * D @ D.T
sigma_low_rank_swag = 0.5 * jnp.diag(sigma_diag_swag) + 0.5 * sigma_low_rank_swag

key, subkey = jax.random.split(key)
samples_diag = mu_swag + jax.random.normal(subkey, (args.nsamples, shape_phi)) * jnp.sqrt(sigma_diag_swag)
samples_low_rank = (mu_swag
                    + jax.random.normal(subkey, (args.nsamples, shape_phi)) @ jnp.linalg.cholesky(sigma_low_rank_swag))

nlpd_diag = nlpd_fn(samples_diag, opt_psi, dataset.ys, dataset.xs)
test_nlpd_diag = nlpd_fn(samples_diag, opt_psi, dataset.test_ys, dataset.test_xs)
print(f'SWAG DIAG NLPD: {nlpd_diag}, test_NLPD: {test_nlpd_diag}')

nlpd_low_rank = nlpd_fn(samples_low_rank, opt_psi, dataset.ys, dataset.xs)
test_nlpd_low_rank = nlpd_fn(samples_low_rank, opt_psi, dataset.test_ys, dataset.test_xs)
print(f'SWAG LOW RANK NLPD: {nlpd_low_rank}, test_NLPD: {test_nlpd_low_rank}')

if args.adam:
    filename = f'./results/moon/swag_adam_{args.id}'
else:
    filename = f'./results/moon/swag_{args.id}'
np.savez(filename,
         samples_diag=samples_diag, samples_low_rank=samples_low_rank, psi=opt_psi,
         nlpd_diag=nlpd_diag, test_nlpd_diag=test_nlpd_diag,
         nlpd_low_rank=nlpd_low_rank, test_nlpd_low_rank=test_nlpd_low_rank)
