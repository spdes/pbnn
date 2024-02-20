"""
Plot the psi trajectory and some distribution samples
"""
import jax
import numpy as np
import matplotlib.pyplot as plt
from pbnn.data.bayesian import Crescent

jax.config.update("jax_enable_x64", True)

# Plot trajectories
methods = ['bfmc_hmc', 'map_hmc', 'ohsmc_rw', 'sgsmc_hmc', 'vb']
labels = ['MC', 'MAP (and SWAG)', 'OHSMC', 'SGSMC', 'VB']
alphas = [1., 1., 1., 1., 1.]
markers = [',', '^', '*', 'x', '|']
styles = ['-', '--', '--', '--', '--']

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 19})

fig, ax = plt.subplots()

for method, marker, style, alpha, label in zip(methods, markers, styles, alphas, labels):
    results = np.load(f'./results/{method}.npz')
    psis = results['psis']
    plt.plot(psis, c='black', linewidth=1.2, linestyle=style,
             marker=marker, markevery=40, markersize=8,
             alpha=alpha, label=label)

ax.grid(linestyle='--', alpha=0.3, which='both')
ax.set_ylabel(r'$\psi$ estimate')
ax.set_xlabel('Epoch')
plt.tight_layout(pad=0.1)
plt.legend(ncols=2, framealpha=0.5)
plt.savefig('crescent-trace.pdf', transparent=True)
plt.show()

# Plot samples
key = jax.random.PRNGKey(666)
data_key, key = jax.random.split(key)
dataset = Crescent(100, data_key, 1.)

grids_x = np.linspace(-2., 2., 1000)
grids_y = np.linspace(-2., 2., 1000)
meshes = np.meshgrid(grids_x, grids_x)
cart = np.dstack(meshes)

fig, axes = plt.subplots()

axes.contour(*meshes, dataset.posterior(cart), levels=5, cmap=plt.cm.binary)

# Plot HMC
# samples = np.load(f'./results/bfmc_hmc.npz')['samples']
# axes.scatter(samples[:, 0], samples[:, 1], s=2, c='black', edgecolors='none')

# Plot OHSMC
samples = np.load(f'./results/ohsmc_rw.npz')['samples']
axes.scatter(samples[:, 0], samples[:, 1], s=4, c='black', edgecolors='none', label='OHSMC samples')

# Plot VB
results = np.load(f'./results/vb.npz')
mean, cov_diag = results['mean'], results['cov_diag']
axes.contour(*meshes,
             jax.vmap(jax.vmap(jax.scipy.stats.multivariate_normal.pdf, in_axes=[0, None, None]),
                      in_axes=[0, None, None])(cart, mean, np.diag(cov_diag)),
             levels=5, linestyles='--', cmap=plt.cm.binary)

axes.plot([], [], c='black', linestyle='-', label=r'True density $p(\phi \mid y_{1:100})$')
axes.plot([], [], c='black', linestyle='--', label='VB approximate')
axes.grid(linestyle='--', alpha=0.3, which='both')
axes.set_xlabel(r'$\phi_0$')
axes.set_ylabel(r'$\phi_1$')
axes.set_xlim(-2, 2)
axes.set_ylim(-2., 1.)
axes.legend(loc='lower center')
plt.tight_layout(pad=0.1)
plt.savefig('crescent-density.pdf', transparent=True)
plt.show()
