"""
Tabulate the moon results.
"""
import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pbnn.utils import accuracy
from pbnn.data.classification import Moons
from pbnn.experiment_settings.nn_configs import moon

import torch
from torchmetrics.functional.classification import multiclass_calibration_error

jax.config.update("jax_enable_x64", True)

methods = ['map_hmc', 'ohsmc', 'sgsmc_hmc', 'swag_adam', 'vb']
labels = ['MAP-HMC', 'OHSMC', 'SGSMC-HMC', 'SWAG', 'VB']
num_mcs = 100

# Plots


plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 17})

fig, axes = plt.subplots(nrows=5, ncols=len(methods), figsize=(14, 10), sharey='row', sharex='col')

query_x1 = np.linspace(-2., 3., 100)
query_x2 = np.linspace(-1.5, 2., 100)
query_xs = np.meshgrid(query_x1, query_x2)
grids = jnp.concatenate([query_xs[0][:, :, None], query_xs[1][:, :, None]], axis=-1).reshape(-1, 2)

for row, mc_id in enumerate(range(0, 100, 20)):
    for i, method, label in zip(range(len(methods)), methods, labels):
        key = jnp.asarray(np.load('./keys_jax.npy')[mc_id])
        data_key, key = jax.random.split(key)

        # Dataset creation
        data_size = 100
        batch_size = 20
        dataset = Moons(n=data_size,
                        rng_state=np.random.RandomState(np.load('./keys_np.npy')[mc_id]),
                        noise=0.3)

        # Reproduce the exact pbnn
        data_key, subkey = jax.random.split(data_key)
        _, _, pbnn_forward_pass = moon(subkey, batch_size)
        pbnn_forward_pass_vmap = jax.jit(jax.vmap(pbnn_forward_pass, in_axes=[None, None, 0]))

        bools = dataset.test_ys[:, 0] == 0
        axes[row, i].scatter(dataset.test_xs[bools, 0], dataset.test_xs[bools, 1], s=20,
                             facecolors='none', edgecolors='black')
        axes[row, i].scatter(dataset.test_xs[~bools, 0], dataset.test_xs[~bools, 1], c='black', s=10)

        filename = f'./results/moon/{method}_{mc_id}.npz'
        results = np.load(filename)
        if 'swag' in method:
            samples = results['samples_low_rank']
        else:
            samples = results['samples']
        for sample in samples[::50]:
            prediction_sample = pbnn_forward_pass(sample, results['psi'], grids).reshape(100, 100)
            axes[row, i].contour(query_x1, query_x2, prediction_sample, levels=[.5], alpha=0.2, colors='black')

        axes[row, i].grid(linestyle='--', alpha=0.3, which='both')

        axes[-1, i].set_xlabel('$x_1$')
        axes[0, i].set_title(f'{label}')
        axes[row, 0].set_ylabel('$x_2$')

axes[0, 0].set_title(rf'MAP-HMC')
axes[0, 1].set_title(rf'OHSMC')
axes[0, 2].set_title(rf'SGSMC-HMC')
axes[0, 3].set_title(rf'SWAG')
axes[0, 4].set_title(rf'VB')

plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('./figs/moon-demo-5.pdf', transparent=True)
plt.show()
