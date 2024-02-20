"""
Tabulate the regression results.
"""
import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from pbnn.data.regression import OneDimGaussian
from pbnn.experiment_settings.nn_configs import syn_regression

jax.config.update("jax_enable_x64", True)

methods = ['map_hmc', 'ohsmc', 'sgsmc_hmc', 'swag_adam', 'vb']
labels = ['MAP-HMC', 'OHSMC', 'SGSMC-HMC', 'SWAG', 'VB']
num_mcs = 100

# Plot results
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 18})

fig, axes = plt.subplots(nrows=5, ncols=len(methods), figsize=(14, 10), sharey='row', sharex='col')

xs = np.linspace(-7., 7., 200)
for row, mc_id in enumerate(range(0, 100, 20)):
    for i, method, label in zip(range(len(methods)), methods, labels):
        key = jnp.asarray(np.load('./keys_jax.npy')[mc_id])
        data_key, key = jax.random.split(key)
        dataset = OneDimGaussian(key=data_key, n=100)

        # Reproduce the exact pbnn
        data_key, subkey = jax.random.split(data_key)
        _, _, pbnn_forward_pass = syn_regression(subkey, 20)
        pbnn_forward_pass = jax.jit(pbnn_forward_pass)

        axes[row, i].plot(xs, dataset.fs(xs), linestyle='--', c='black', label='True function')
        axes[row, i].scatter(dataset.test_xs, dataset.test_ys, c='black', s=2, label='Observations')

        filename = f'./results/regression/{method}_{mc_id}.npz'
        results = np.load(filename)
        if 'swag' in method:
            samples = results['samples_low_rank']
        else:
            samples = results['samples']
        for sample in samples[::50]:
            prediction_sample = pbnn_forward_pass(sample, results['psi'], xs[:, None])
            axes[row, i].plot(xs, prediction_sample, linewidth=1, c='black', alpha=0.2)

        axes[row, i].grid(linestyle='--', alpha=0.3, which='both')
        axes[row, i].set_ylim([-6, 6])
        axes[row, i].set_yticks([-5, 0, 5])

        axes[0, i].set_title(f'{label}')
        axes[-1, i].set_xlabel('$x$')
        axes[row, 0].set_ylabel('$y$')

axes[0, 0].set_title(rf'MAP-HMC')
axes[0, 1].set_title(rf'OHSMC')
axes[0, 2].set_title(rf'SGSMC-HMC')
axes[0, 3].set_title(rf'SWAG')
axes[0, 4].set_title(rf'VB')

plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('./figs/regression-demo-5.pdf', transparent=True)
plt.show()
