"""
Parameter estimation and posterior computation on the crescent model.

Using VB.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from pbnn.data.bayesian import Crescent
from pbnn.solvers import variational_bayes
from functools import partial

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


# Make mean-field approximate posterior
@partial(jax.vmap, in_axes=[0, None])
def log_pdf_approx_posterior(phi, _theta):
    m, log_v = _theta[:2], _theta[2:]
    return jnp.sum(jax.scipy.stats.norm.logpdf(phi, m, jnp.exp(0.5 * log_v)))


def approx_posterior_sampler(_theta, _key, _nsamples):
    m, log_v = _theta[:2], _theta[2:]
    return m + jax.random.normal(_key, (_nsamples, 2)) * jnp.exp(0.5 * log_v)


# Make loss function
log_prior_pdf = jax.vmap(dataset.log_prior_pdf, in_axes=[0])
elbo = variational_bayes(log_cond_pdf_likelihood_vmap, log_prior_pdf, log_pdf_approx_posterior,
                         lambda u, v: approx_posterior_sampler(u, v, 100),
                         data_size=data_size)


def loss_fn(_param, _key, _ys, _xs):
    _psi, _theta = _param[0], _param[1:]
    return -elbo(_psi, _theta, _key, _ys, _xs)


# MAP optimisation
@jax.jit
def opt_step_kernel(_param, _opt_state, _ys, _xs, _key):
    _loss, grad = jax.value_and_grad(loss_fn)(_param, _key, _ys, _xs)
    updates, _opt_state = optimiser.update(grad, _opt_state, _param)
    _param = optax.apply_updates(_param, updates)
    return _param, _opt_state, _loss


# Optimisation setup
nepochs = 200
schedule = optax.exponential_decay(1e-1, 10, 0.96)
optimiser = optax.adam(learning_rate=schedule)
param = jnp.array([0.1, 0., 0., 0., 0.])
opt_state = optimiser.init(param)
psis = np.zeros((nepochs,))

# Run
loss = 0.
for i in range(nepochs):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        key, subkey = jax.random.split(key)
        param, opt_state, loss = opt_step_kernel(param, opt_state, ys, xs, subkey)
    psis[i] = param[0]
    print(f'Epoch: {i}, loss: {loss}, psi: {param[0]}')

# Unpack learnt values
opt_psi, mean, cov_diag = param[0], param[1:3], jnp.exp(0.5 * param[3:5])

# Save results
np.savez(f'./results/vb', psis=psis, mean=mean, cov_diag=cov_diag)

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

plt.plot(psis)
plt.show()

# Plot densities
grids_x = jnp.linspace(-2., 2., 1000)
grids_y = jnp.linspace(-2., 2., 1000)
meshes = jnp.meshgrid(grids_x, grids_x)
cart = jnp.dstack(meshes)

plt.contour(*meshes, dataset.posterior(cart), levels=5, cmap=plt.cm.binary, label='Posterior density')
plt.contour(*meshes,
            jax.vmap(jax.vmap(jax.scipy.stats.multivariate_normal.pdf, in_axes=[0, None, None]),
                     in_axes=[0, None, None])(cart, mean, np.diag(cov_diag)),
            levels=5, cmap=plt.cm.Blues, label='VB')
plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.25, 0.5)
plt.legend()

plt.tight_layout(pad=0.1)
plt.show()
