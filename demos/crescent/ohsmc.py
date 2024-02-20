"""
Parameter estimation and posterior computation on the crescent model.

Using open-horizon SMC.
"""
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from pbnn.data.bayesian import Crescent
from pbnn.solvers import smc_kernel_log, stratified
from pbnn.markov_kernels import make_random_walk, make_adaptive_random_walk

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

# SMC setup
nsamples = 1000

transition = 'rw'

if transition == 'rw':
    transition_sampler = make_random_walk(1e-3)
else:
    transition_sampler = make_adaptive_random_walk(0.1, log=True, whitened_cov=True)


def resampling(_us, _ws, _key, _dummy):
    return stratified(_us, _ws, _key)


@jax.jit
def ohsmc(_samples, _log_weights, _psi, _opt_state, _key, _ys, _xs):
    """Open-horizon SMC step.
    """
    _samples, _log_weights, _ = smc_kernel_log(_samples, _log_weights, _ys, _xs, transition_sampler, 1.,
                                               log_cond_pdf_likelihood_vmap, _psi, _key,
                                               resampling_method=stratified, resampling_threshold=1.)
    grad = -data_size / batch_size * jnp.dot(jnp.exp(_log_weights),
                                             grad_log_cond_pdf_likelihood(_ys, _samples, _xs, _psi))
    updates, _opt_state = optimiser.update(grad, _opt_state, _psi)
    _psi = optax.apply_updates(_psi, updates)
    return _samples, _log_weights, _psi, _opt_state


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
        key, subkey = jax.random.split(key)
        samples, log_weights, psi, opt_state = ohsmc(samples, log_weights, psi, opt_state, subkey, ys, xs)
    psis[i] = psi
    print(f'Epoch: {i}, psi: {psi}')

# Save results
np.savez(f'./results/ohsmc_{transition}', psis=psis, samples=samples)

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

plt.plot(psis)
plt.show()

grids_x = jnp.linspace(-2., 2., 1000)
grids_y = jnp.linspace(-2., 2., 1000)
meshes = jnp.meshgrid(grids_x, grids_x)
cart = jnp.dstack(meshes)

plt.contour(*meshes, dataset.posterior(cart), levels=5, cmap=plt.cm.binary, label='Posterior density')
plt.scatter(samples[:, 0], samples[:, 1], s=1, c='black', edgecolors='none', alpha=0.8, label='OHSMC samples')
plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.25, 0.5)
plt.legend()

plt.tight_layout(pad=0.1)
plt.show()
