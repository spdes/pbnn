"""
Run a classification experiment on the moon dataset using the OHSMC algorithm.
"""
import argparse
import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from pbnn.data.classification import Moons
from pbnn.solvers import smc_kernel_log, stratified
from pbnn.markov_kernels import make_random_walk
from pbnn.nn import make_pbnn_likelihood
from pbnn.experiment_settings.nn_configs import moon

jax.config.update("jax_enable_x64", True)

# Parse arguments
parser = argparse.ArgumentParser(description='Synthetic classification using OH-SMC.')
parser.add_argument('--nsamples', type=int, default=1000, help='The number of SMC samples.')
parser.add_argument('--rw_var', type=float, default=1e-2, help='The random walk transition variance.')
parser.add_argument('--id', type=int, default=0, help='The Monte Carlo run id (0 - 1000).')
parser.add_argument('--data_size', type=int, default=100, help='Data size.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=100, help='The maximum number of iterations.')
args = parser.parse_args()

print('Running moon classification with OHSMC.')

# Random key seed
# Separate the key branch for data and algorithm
key = jax.random.PRNGKey(666)
data_key, key = jax.random.split(key)

# Dataset creation
data_size = args.data_size
batch_size = args.batch_size
dataset = Moons(n=data_size,
                rng_state=np.random.RandomState(666),
                noise=0.3)

# Define the pBNN
data_key, subkey = jax.random.split(data_key)
pbnn_phi, pbnn_psi, pbnn_forward_pass = moon(subkey, batch_size)
shape_phi, shape_psi = pbnn_phi[0].shape[0], pbnn_psi[0].shape[0]

log_cond_pdf_likelihood, _, _ = make_pbnn_likelihood(pbnn_forward_pass, likelihood_type='bernoulli')
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


# Optimisation setup
schedule = optax.exponential_decay(1e-2, data_size // batch_size, 0.96)
optimiser = optax.adam(learning_rate=schedule)
param = pbnn_psi[0]
opt_state = optimiser.init(param)

# Optimisation loop
key, subkey = jax.random.split(key)
samples = m0 + jnp.sqrt(v0) * jax.random.normal(subkey, (nsamples, shape_phi))
log_weights = -jnp.log(nsamples) * jnp.ones((nsamples,))
psi = pbnn_psi[0]
for i in range(args.max_epochs):
    print(f'Epoch {i} / {args.max_epochs}')
    data_key, _ = jax.random.split(data_key)
    dataset.init_enumeration(data_key, batch_size)
    for j in range(int(data_size / batch_size)):
        xs, ys = dataset.enumerate_subset(j)
        key, subkey = jax.random.split(key)
        samples, log_weights, psi, opt_state = ohsmc(samples, log_weights, psi, opt_state, subkey, ys, xs)

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 17})

fig, ax = plt.subplots()

query_x1 = np.linspace(-2., 3., 100)
query_x2 = np.linspace(-1.5, 2., 100)
query_xs = np.meshgrid(query_x1, query_x2)
grids = jnp.concatenate([query_xs[0][:, :, None], query_xs[1][:, :, None]], axis=-1).reshape(-1, 2)

bools = dataset.test_ys[:, 0] == 0
ax.scatter(dataset.test_xs[bools, 0], dataset.test_xs[bools, 1], s=20,
           facecolors='none', edgecolors='black', label='Class 1')
ax.scatter(dataset.test_xs[~bools, 0], dataset.test_xs[~bools, 1], c='black', s=10, label='Class 2')

for sample in samples[::50]:
    prediction_sample = pbnn_forward_pass(sample, psi, grids).reshape(100, 100)
    ax.contour(query_x1, query_x2, prediction_sample, levels=[.5], alpha=0.2, colors='black')

ax.grid(linestyle='--', alpha=0.3, which='both')
ax.legend()

ax.set_xlabel('$x_1$')
ax.set_title('Classification (grey lines are classifier samples)')
ax.set_ylabel('$x_2$')

plt.tight_layout(pad=0.1)
plt.savefig('../figs/moon_ohsmc.svg', transparent=True)
plt.show()
