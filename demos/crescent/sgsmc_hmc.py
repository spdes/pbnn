"""
Parameter estimation and posterior computation on the crescent model.

Using stochastic-gradient SMC and HMC.
"""
import math
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from pbnn.data.bayesian import Crescent
from pbnn.solvers import chsmc, stratified, hmc
from pbnn.markov_kernels import make_random_walk, make_adaptive_random_walk, make_pbnn_rwmrh

jax.config.update("jax_enable_x64", True)

# Random key seed
# Separate the key branch for data and algorithm
key = jax.random.PRNGKey(666)
data_key, key = jax.random.split(key)

# Dataset creation
data_size = 100
batch_size = 10
true_psi = 1.
dataset = Crescent(data_size, data_key, true_psi)


# Define the likelihood model
def log_cond_pdf_likelihood(_ys, _sample, _, _psi):
    return dataset.log_cond_pdf_likelihood(_ys, _sample, _psi)


log_cond_pdf_likelihood_vmap = jax.vmap(log_cond_pdf_likelihood, in_axes=[None, 0, None, None])
grad_log_cond_pdf_likelihood = jax.vmap(jax.grad(log_cond_pdf_likelihood, argnums=3), in_axes=[None, 0, None, None])

# Prior definition
m0 = dataset.m
cov0 = dataset.cov


def log_pdf_prior(phi):
    return jnp.sum(jax.scipy.stats.norm.logpdf(phi, 0., math.sqrt(1.)))


# SMC setup
nsamples = 1000


def log_posterior(_phi, _args):
    _psi, _inflated_y, _inflated_x = _args
    return log_cond_pdf_likelihood(_inflated_y, _phi, _inflated_x, _psi) + dataset.log_prior_pdf(_phi)


transition_sampler = make_pbnn_rwmrh(log_posterior, 1e-3, 10)


def resampling(_us, _ws, _key, _dummy):
    return stratified(_us, _ws, _key)


@jax.jit
def smc(_samples, _log_weights, _psi, _opt_state, _key, _ys, _xs, _inflated_ys, _inflated_xs):
    _samples = m0 + jax.random.normal(_key, (nsamples, 2)) @ jnp.linalg.cholesky(cov0)
    _log_weights = -jnp.log(nsamples) * jnp.ones((nsamples,))
    _, _subkey = jax.random.split(_key)
    _samples, _log_weights, _nell = chsmc(_samples, _log_weights, _ys, _xs, _inflated_ys, _inflated_xs,
                                          transition_sampler,
                                          log_cond_pdf_likelihood_vmap, _psi, _subkey, True,
                                          resampling_method=stratified, resampling_threshold=1.)
    grad = -data_size / batch_size * jnp.dot(jnp.exp(_log_weights),
                                             grad_log_cond_pdf_likelihood(_ys, _samples, _xs, _psi))
    updates, _opt_state = optimiser.update(grad, _opt_state, _psi)
    _psi = optax.apply_updates(_psi, updates)
    return _samples, _log_weights, _psi, _opt_state, _nell


# Optimisation setup
nepochs = 200
schedule = optax.exponential_decay(1e-1, 10, 0.96)
optimiser = optax.adam(learning_rate=schedule)
psi = 0.1
opt_state = optimiser.init(psi)
psis = np.zeros((nepochs,))

# Run
key, subkey = jax.random.split(key)
samples = m0 + jax.random.normal(subkey, (nsamples, 2)) @ jnp.linalg.cholesky(cov0)
log_weights = -jnp.log(nsamples) * jnp.ones((nsamples,))
for i in range(nepochs):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        inflated_xs, inflated_ys = dataset.inflate_nan(xs, ys)
        key, subkey = jax.random.split(key)
        samples, log_weights, psi, opt_state, nell = smc(samples, log_weights, psi, opt_state, subkey,
                                                         ys, xs, inflated_ys, inflated_xs)
    psis[i] = psi
    print(f'Epoch: {i}, loss: {nell * data_size / batch_size}, psi: {psi}')


# Do HMC
def log_posterior(phi):
    return log_cond_pdf_likelihood(dataset.ys, phi, dataset.xs, psi) + dataset.log_prior_pdf(phi)


init_sample = jnp.zeros((2,))
dt = 1e-2
integration_steps = 100
inv_mass = jnp.ones_like(init_sample)

nsamples = 3000
burn_in = 2000
key, _ = jax.random.split(key)
samples = hmc(log_posterior, init_sample, dt, integration_steps, inv_mass, nsamples, key, verbose=1)

# Save results
np.savez(f'./results/sgsmc_hmc', psis=psis, samples=samples)

# Plot trace for debug
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

plt.plot(samples[:, 0])
plt.show()

# Plot samples
grids_x = jnp.linspace(-2., 2., 1000)
grids_y = jnp.linspace(-2., 2., 1000)
meshes = jnp.meshgrid(grids_x, grids_x)
cart = jnp.dstack(meshes)

plt.contour(*meshes, dataset.posterior(cart), levels=5, cmap=plt.cm.binary, label='Posterior density')
plt.scatter(samples[:, 0], samples[:, 1], s=1, c='black', edgecolors='none', alpha=0.8, label='HMC samples')
plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.25, 0.5)
plt.legend()

plt.tight_layout(pad=0.1)
plt.show()
