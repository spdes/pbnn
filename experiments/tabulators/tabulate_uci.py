"""
Tabulate UCI
"""
import jax
import jax.numpy as jnp
import numpy as np
from pbnn.data.uci import UCIEnum
from pbnn.experiment_settings.nn_configs import uci
from pbnn.utils import accuracy
import torch
from torchmetrics.functional.classification import multiclass_calibration_error

jax.config.update("jax_enable_x64", True)

datasets_reg = ['boston', 'concrete', 'energy', 'kin8', 'naval', 'yacht', 'power']
datasets_cla = ['australian', 'cancer', 'ionosphere', 'glass', 'satellite']

methods = ['map_hmc', 'ohsmc_0.01', 'ohsmc_ou', 'sgsmc_hmc', 'swag_adam', 'vb']
num_mcs = 10

# This tabulates the regression results
for dataset_name in datasets_reg:
    for method in methods:
        nlpds = np.zeros((num_mcs,))
        rmses = np.zeros((num_mcs,))
        for i in range(num_mcs):
            filename = f'./results/uci/{dataset_name}_{method}_{i}.npz'
            results = np.load(filename)

            if 'swag' in method:
                nlpd = results['test_nlpd_low_rank']
            else:
                nlpd = results['test_nlpd']

            nlpds[i] = nlpd

            # Compute RMSE
            key = jnp.asarray(np.load('./keys_jax.npy')[i])
            data_key, key = jax.random.split(key)

            # Dataset creation
            dataset = UCIEnum[dataset_name].value('./data', np.random.RandomState(np.load('./keys_np.npy')[i]))
            data_size = dataset.n
            batch_size = 20

            # Define the pBNN
            data_key, subkey = jax.random.split(data_key)
            pbnn_phi, pbnn_psi, pbnn_forward_pass = uci(subkey, batch_size, dataset.input_dim, 1)
            if 'swag' in method:
                preds = jax.vmap(pbnn_forward_pass,
                                 in_axes=[0, None, None])(results['samples_low_rank'], results['psi'],
                                                          dataset.test_xs)[:, :, 0]
            else:
                preds = jax.vmap(pbnn_forward_pass,
                                 in_axes=[0, None, None])(results['samples'], results['psi'], dataset.test_xs)[:, :, 0]
            rmses[i] = jnp.mean(jnp.sqrt(jnp.mean((preds - dataset.test_ys[:, 0]) ** 2, axis=1)))

        print(f'{dataset_name} | {method} | nlpd: {np.mean(nlpds):.4f} | std. {np.std(nlpds):.4f} | '
              f'rmse {np.mean(rmses):.4f} | std. {np.std(rmses):.4f}')

# This tabulates the classification results
for dataset_name in datasets_cla:
    for method in methods:
        nlpds = np.zeros((num_mcs,))
        eces = np.zeros((num_mcs, 1000))
        accs = np.zeros((num_mcs, 1000))
        for i in range(num_mcs):
            filename = f'./results/uci/{dataset_name}_{method}_{i}.npz'
            results = np.load(filename)

            if 'swag' in method:
                nlpd = results['test_nlpd_low_rank']
            else:
                nlpd = results['test_nlpd']

            nlpds[i] = nlpd

            # Compute expected calibration error
            key = jnp.asarray(np.load('./keys_jax.npy')[i])
            data_key, key = jax.random.split(key)

            # Dataset creation
            dataset = UCIEnum[dataset_name].value('./data', np.random.RandomState(np.load('./keys_np.npy')[i]))
            data_size = dataset.n
            batch_size = 20

            # Define the pBNN
            data_key, subkey = jax.random.split(data_key)
            _, _, pbnn_forward_pass = uci(subkey, batch_size, dataset.input_dim, dataset.nlabel)
            pbnn_forward_pass_vmap = jax.vmap(pbnn_forward_pass, in_axes=[0, None, None])
            if 'swag' in method:
                samples = results['samples_low_rank']
            else:
                samples = results['samples']
            psi = results['psi']

            true_ys = dataset.test_ys.argmax(1)  # (s, )
            for j, sample in enumerate(samples):
                predictions = jax.nn.softmax(pbnn_forward_pass(sample, psi, dataset.test_xs))
                eces[i, j] = multiclass_calibration_error(torch.tensor(np.array(predictions)),
                                                          torch.tensor(np.array(true_ys)),
                                                          num_classes=dataset.nlabel, n_bins=15, norm='l1')
                accs[i, j] = accuracy(predictions, dataset.test_ys)

        ave_eces = np.mean(eces, axis=1)
        ave_accs = np.mean(accs, axis=1)
        print(f'{dataset_name} | {method} | nlpd: {np.mean(nlpds):.4f} | std. {np.std(nlpds):.4f} '
              f'| ece {np.mean(eces):.4f} | std. {np.std(eces):.4f} '
              f'| acc {np.mean(ave_accs):.4f} | std. {np.std(ave_accs):.4f}')
