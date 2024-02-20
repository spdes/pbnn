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

for method in methods:
    nlpds = np.zeros((num_mcs, ))
    eces = np.zeros((num_mcs,))
    accs = np.zeros((num_mcs, 1000))
    for i in range(num_mcs):
        filename = f'./results/moon/{method}_{i}.npz'
        results = np.load(filename)

        if 'swag' in method:
            nlpd = results['test_nlpd_low_rank']
        else:
            nlpd = results['test_nlpd']

        nlpds[i] = nlpd

        # Compute ECE
        key = jnp.asarray(np.load('./keys_jax.npy')[i])
        data_key, key = jax.random.split(key)

        # Dataset creation
        data_size = 100
        batch_size = 20
        dataset = Moons(n=data_size,
                        rng_state=np.random.RandomState(np.load('./keys_np.npy')[i]),
                        noise=0.3)

        # Reproduce the exact pbnn
        data_key, subkey = jax.random.split(data_key)
        _, _, pbnn_forward_pass = moon(subkey, batch_size)
        pbnn_forward_pass_vmap = jax.jit(jax.vmap(pbnn_forward_pass, in_axes=[0, None, None]))

        if 'swag' in method:
            samples = results['samples_low_rank']
        else:
            samples = results['samples']
        psi = results['psi']

        _eces = np.zeros((1000,))
        true_ys = dataset.test_ys[:, 0]
        preds = jax.nn.sigmoid(pbnn_forward_pass_vmap(samples, psi, dataset.test_xs))
        preds = jnp.concatenate([1 - preds, preds], axis=-1)
        for j in range(1000):
            ece_per_sample = multiclass_calibration_error(torch.tensor(np.array(preds[j])),
                                                          torch.tensor(np.array(true_ys)),
                                                          num_classes=2, norm='l1')
            _eces[j] = ece_per_sample
            accs[i, j] = accuracy(preds[j], jax.nn.one_hot(dataset.test_ys, num_classes=2, axis=1)[:, :, 0])
        eces[i] = np.mean(_eces)

    ave_acc = np.mean(accs, axis=1)
    print(f'{method} | nlpd: {np.mean(nlpds)} | std. {np.std(nlpds)} | ece {np.mean(eces)} | std. {np.std(eces)} | acc {np.mean(ave_acc)} | std. {np.std(ave_acc)}')


# Plots
mc_id = 1
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

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 17})

fig, axes = plt.subplots(ncols=len(methods), figsize=(18, 3.8), sharey='row')

query_x1 = np.linspace(-2., 3., 100)
query_x2 = np.linspace(-1.5, 2., 100)
query_xs = np.meshgrid(query_x1, query_x2)
grids = jnp.concatenate([query_xs[0][:, :, None], query_xs[1][:, :, None]], axis=-1).reshape(-1, 2)

for i, method, label in zip(range(len(methods)), methods, labels):
    bools = dataset.test_ys[:, 0] == 0
    axes[i].scatter(dataset.test_xs[bools, 0], dataset.test_xs[bools, 1], s=20,
                    facecolors='none', edgecolors='black')
    axes[i].scatter(dataset.test_xs[~bools, 0], dataset.test_xs[~bools, 1], c='black', s=10)

    filename = f'./results/moon/{method}_{mc_id}.npz'
    results = np.load(filename)
    if 'swag' in method:
        samples = results['samples_low_rank']
    else:
        samples = results['samples']
    for sample in samples[::50]:
        prediction_sample = pbnn_forward_pass(sample, results['psi'], grids).reshape(100, 100)
        axes[i].contour(query_x1, query_x2, prediction_sample, levels=[.5], alpha=0.2, colors='black')

    axes[i].grid(linestyle='--', alpha=0.3, which='both')
    axes[i].set_xlabel('$x_1$')
    axes[i].set_title(f'{label}')

axes[0].set_title(rf'MAP-HMC (ECE=0.07)')
axes[1].set_title(rf'OHSMC (ECE=0.06)')
axes[2].set_title(rf'SGSMC-HMC (ECE=0.08)')
axes[3].set_title(rf'SWAG (ECE=0.07)')
axes[4].set_title(rf'VB (ECE=0.08)')

# axes[0].legend()
axes[0].set_ylabel('$x_2$')
plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0)
plt.savefig('./figs/moon-demo.pdf', transparent=True)
plt.show()

# map_hmc | nlpd: 0.28458915031946785 | std. 0.06157948368695749 | ece 0.07933550344115124 | std. 0.015781067923238894 | acc 0.8764746999999998 | std. 0.02490556831132348
# ohsmc | nlpd: 0.28769203056433956 | std. 0.07729477038405456 | ece 0.06988573669791222 | std. 0.015081110261647704 | acc 0.8865195 | std. 0.025644520053024988
# sgsmc_hmc | nlpd: 0.3232742613179439 | std. 0.08499034353609566 | ece 0.0824401471280493 | std. 0.016911562272953838 | acc 0.8674080000000001 | std. 0.03214430954305911
# swag_adam | nlpd: 0.31220200507802154 | std. 0.06505322120157615 | ece 0.07867880512272939 | std. 0.016337777633062732 | acc 0.8659739999999998 | std. 0.03542216413490289
# vb | nlpd: 0.2914135099585931 | std. 0.05523897356081153 | ece 0.08328043284362183 | std. 0.013171608875518067 | acc 0.8648053 | std. 0.030495366417047694