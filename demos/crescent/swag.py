"""
Parameter estimation and posterior computation on the crescent model.

Using SWAG. Implementation by Sebastian.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from pbnn.data.bayesian import Crescent
from pbnn.solvers import maximum_a_posteriori

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

# Make loss function
ell = maximum_a_posteriori(log_cond_pdf_likelihood, dataset.log_prior_pdf, data_size=data_size)


def loss_fn(_param, _ys, _xs):
    _phi, psi = _param[:2], _param[-1]
    return -ell(_phi, psi, _ys, _xs)


# MAP optimisation
@jax.jit
def opt_step_kernel(_param, _opt_state, _ys, _xs):
    _loss, grad = jax.value_and_grad(loss_fn)(_param, _ys, _xs)
    updates, _opt_state = optimiser.update(grad, _opt_state, _param)
    _param = optax.apply_updates(_param, updates)
    return _param, _opt_state, _loss


# Optimisation setup
nepochs = 200
schedule = optax.exponential_decay(1e-1, 10, 0.96)
optimiser = optax.adam(learning_rate=schedule)
nswags = 100
param = jnp.array([0., 0., 0.1])
opt_state = optimiser.init(param)
psis = np.zeros((nepochs,))

# Pre-train
loss = 0.
for i in range(nepochs):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        param, opt_state, loss = opt_step_kernel(param, opt_state, ys, xs)
    psis[i] = param[-1]
    print(f'Epoch: {i}, loss: {loss}, psi: {param[-1]}')

# Unpack learnt values
opt_phi, opt_psi = param[:2], param[-1]

# train
K = 20
phis = np.zeros((nepochs, 2))
phi_bar = opt_phi
phi2_bar = opt_phi ** 2
D = []
optimiser = optax.sgd(learning_rate=1e-3)
opt_state = optimiser.init(param)
for i in range(nswags):
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        param, opt_state, loss = opt_step_kernel(param, opt_state, ys, xs)

    phi = param[:2]
    phi_bar = (i * phi_bar + phi) / (i + 1)
    phi2_bar = (i * phi2_bar + phi ** 2) / (i + 1)
    D.append(phi - phi_bar)
    D = D[-K:]

    phis[i] = phi
    print(f'Epoch: {i}, loss: {loss}, psi: {param[-1]}')

mu_swag = phi_bar
sigma_diag_swag = phi2_bar - phi_bar ** 2
D = jnp.array(D).T
sigma_low_rank_swag = 1. / (K + 1) * D @ D.T

# Save results
np.savez(f'./results/swag', psis=psis, phis=phis,
         mu_swag=mu_swag, sigma_diag_swag=sigma_diag_swag, sigma_low_rank_swag=sigma_low_rank_swag)

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

# Plot densities
grids_x = jnp.linspace(-2., 2., 1000)
grids_y = jnp.linspace(-2., 2., 1000)
meshes = jnp.meshgrid(grids_x, grids_x)
cart = jnp.dstack(meshes)

plt.contour(*meshes, dataset.posterior(cart), levels=5, cmap=plt.cm.binary, label='Posterior density')
plt.contour(*meshes,
            jax.vmap(jax.vmap(jax.scipy.stats.multivariate_normal.pdf, in_axes=[0, None, None]),
                     in_axes=[0, None, None])(cart, mu_swag, np.diag(sigma_diag_swag)),
            levels=5, cmap=plt.cm.Blues, label='SWAG-DIAG density')
plt.contour(*meshes,
            jax.vmap(jax.vmap(jax.scipy.stats.multivariate_normal.pdf, in_axes=[0, None, None]),
                     in_axes=[0, None, None])(cart, mu_swag, sigma_low_rank_swag),
            levels=5, cmap=plt.cm.binary, label='SWAG-DIAG density')
plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.25, 0.5)
plt.legend()

plt.tight_layout(pad=0.1)
plt.show()
