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

for method in methods:
    nlpds = np.zeros((num_mcs,))
    rmses = np.zeros((num_mcs,))
    for i in range(num_mcs):
        filename = f'./results/regression/{method}_{i}.npz'
        results = np.load(filename)

        if 'swag' in method:
            nlpd = results['test_nlpd_low_rank']
        else:
            nlpd = results['test_nlpd']

        nlpds[i] = nlpd

        # Compute rmse
        # Reproduce the exact random seed
        key = jnp.asarray(np.load('./keys_jax.npy')[i])
        data_key, key = jax.random.split(key)
        dataset = OneDimGaussian(key=data_key, n=100)

        # Reproduce the exact pbnn
        data_key, subkey = jax.random.split(data_key)
        _, _, pbnn_forward_pass = syn_regression(subkey, 20)

        # Compute
        if 'swag' in method:
            preds = jax.vmap(pbnn_forward_pass,
                             in_axes=[0, None, None])(results['samples_low_rank'], results['psi'],
                                                      dataset.test_xs)[:, :, 0]
        else:
            preds = jax.vmap(pbnn_forward_pass,
                             in_axes=[0, None, None])(results['samples'], results['psi'], dataset.test_xs)[:, :, 0]
        true_fs = dataset.fs(dataset.test_xs)[:, 0]
        rmses[i] = jnp.mean(jnp.sqrt(jnp.mean((preds - true_fs) ** 2, axis=1)))

    print(f'{method} | nlpd: {np.mean(nlpds)} | std. {np.std(nlpds)} | rmse: {np.mean(rmses)} | std. {np.std(rmses)}')

# Plot results
mc_id = 0
key = jnp.asarray(np.load('./keys_jax.npy')[mc_id])
data_key, key = jax.random.split(key)
dataset = OneDimGaussian(key=data_key, n=100)

# Reproduce the exact pbnn
data_key, subkey = jax.random.split(data_key)
_, _, pbnn_forward_pass = syn_regression(subkey, 20)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 18})

fig, axes = plt.subplots(ncols=len(methods), figsize=(18, 3.8), sharey='row')

xs = np.linspace(-7., 7., 200)
for i, method, label in zip(range(len(methods)), methods, labels):
    axes[i].plot(xs, dataset.fs(xs), linestyle='--', c='black', label='True function')
    axes[i].scatter(dataset.test_xs, dataset.test_ys, c='black', s=2, label='Observations')

    filename = f'./results/regression/{method}_{mc_id}.npz'
    results = np.load(filename)
    if 'swag' in method:
        samples = results['samples_low_rank']
    else:
        samples = results['samples']
    for sample in samples[::50]:
        prediction_sample = pbnn_forward_pass(sample, results['psi'], xs[:, None])
        axes[i].plot(xs, prediction_sample, linewidth=1, c='black', alpha=0.2)

    axes[i].grid(linestyle='--', alpha=0.3, which='both')
    axes[i].set_xlabel('$x$')
    axes[i].set_title(f'{label}')
    axes[i].set_ylim([-6, 6])
    axes[i].set_yticks([-5, 0, 5])

# axes[0].legend()
axes[0].set_title(rf'MAP-HMC (NLPD=1.53)')
axes[1].set_title(rf'OHSMC (NLPD=1.49)')
axes[2].set_title(rf'SGSMC-HMC (NLPD=1.65)')
axes[3].set_title(rf'SWAG (NLPD=1.71)')
axes[4].set_title(rf'VB (NLPD=2.13)')
axes[0].set_ylabel('$y$')
plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0)
plt.savefig('./figs/regression-demo.pdf', transparent=True)
plt.show()
