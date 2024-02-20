"""
Parameter estimation and posterior computation on the crescent model.

Using brute-force Monte Carlo MLE lower bound and HMC sampling.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from pbnn.data.bayesian import Crescent
from pbnn.solvers import maximum_likelihood, hmc

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


def prior_sampler(_key):
    return dataset.m + jax.random.normal(_key, (10000, 2)) @ jnp.linalg.cholesky(dataset.cov)


ell_lb = maximum_likelihood(log_cond_pdf_likelihood_vmap, prior_sampler, data_size)


# Make loss function
def loss_fn(_psi, _ys, _xs, _key):
    return -ell_lb(_psi, _key, _ys, _xs)


# Optax
@jax.jit
def opt_step_kernel(_param, _opt_state, _ys, _xs, _key):
    _loss, grad = jax.value_and_grad(loss_fn)(_param, _ys, _xs, _key)
    updates, _opt_state = optimiser.update(grad, _opt_state, _param)
    _param = optax.apply_updates(_param, updates)
    return _param, _opt_state, _loss


# Optimisation setup
nepochs = 200
schedule = optax.exponential_decay(1e-1, 10, 0.96)
optimiser = optax.adam(learning_rate=schedule)
psi = 0.1
opt_state = optimiser.init(psi)
psis = np.zeros((nepochs,))

# Run
loss = 0.
for i in range(nepochs):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        key, subkey = jax.random.split(key)
        psi, opt_state, loss = opt_step_kernel(psi, opt_state, ys, xs, subkey)
    psis[i] = psi
    print(f'Epoch: {i}, loss: {loss}, psi: {psi}')

# HMC
def log_posterior(phi):
    return log_cond_pdf_likelihood(dataset.ys, phi, dataset.xs, psi) + dataset.log_prior_pdf(phi)


init_sample = jnp.zeros((2, ))
dt = 1e-2
integration_steps = 100
inv_mass = jnp.ones_like(init_sample)

nsamples = 3000
burn_in = 2000
key, _ = jax.random.split(key)
samples = hmc(log_posterior, init_sample, dt, integration_steps, inv_mass, nsamples, key, verbose=1)
samples = samples[burn_in:]

# Save results
np.savez(f'./results/bfmc_hmc', psis=psis, samples=samples)

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
