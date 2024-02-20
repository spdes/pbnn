"""
Tabulate MNIST
"""
import numpy as np
import jax
import jax.numpy as jnp
from pbnn.data.classification import MNIST
from pbnn.experiment_settings.nn_configs import mnist
import torch
from torchmetrics.functional.classification import multiclass_calibration_error
from pbnn.utils import accuracy

jax.config.update("jax_enable_x64", True)

methods = ['ohsmc', 'swag_adam', 'vb']
which_swag = 'diag'  # 'low_rank'
num_mcs = 10

# Compute NLPD
for method in methods:
    nlpds = np.zeros((num_mcs,))
    for i in range(num_mcs):
        filename = f'./results/mnist/{method}_{i}.npz'
        results = np.load(filename)

        if 'swag' in method:
            nlpd = results[f'test_nlpd_{which_swag}']
        else:
            nlpd = results['test_nlpd']

        nlpds[i] = nlpd
    print(f'{method} | nlpd: {np.mean(nlpds)} | std. {np.std(nlpds)}')

# Compute ECE and acc
for method in methods:
    eces = np.zeros((num_mcs, 100))
    accs = np.zeros((num_mcs, 100))

    for mc_id in range(num_mcs):
        # Reproduce the exact random seed and pBNN config for each MC run.
        key = jnp.asarray(np.load('./keys_jax.npy')[mc_id])
        data_key, key = jax.random.split(key)

        dataset = MNIST('./data/mnist.npz', data_key)
        data_size = dataset.n
        batch_size = 100

        data_key, subkey = jax.random.split(data_key)
        _, _, pbnn_forward_pass = mnist(subkey, batch_size)

        # Load the learnt params
        filename = f'./results/mnist/{method}_{mc_id}.npz'
        results = np.load(filename)
        if 'swag' in method:
            samples = results['samples_low_rank']
        else:
            samples = results['samples']
        psi = results['psi']

        true_ys = dataset.test_ys.argmax(1)  # Shape is (s, 1)
        for i, sample in enumerate(samples[:100]):
            print(f'{method} mc_id={mc_id} i={i}')
            predictions = jax.nn.softmax(pbnn_forward_pass(sample, psi, dataset.test_xs))
            eces[mc_id, i] = multiclass_calibration_error(torch.tensor(np.array(predictions)),
                                                          torch.tensor(np.array(true_ys)),
                                                          num_classes=10, n_bins=15, norm='l1')
            accs[mc_id, i] = accuracy(predictions, dataset.test_ys)
            np.save(f'./results/mnist/preds_{method}_{mc_id}_{i}', predictions)

    ave_eces = np.mean(eces, axis=1)
    ave_accs = np.mean(accs, axis=1)
    print(f'{method}: averaged ECE {np.mean(ave_eces)} | std. {np.std(ave_eces)}')
    print(f'{method}: averaged acc {np.mean(ave_accs)} | std. {np.std(ave_accs)}')
    np.savez(f'./results/mnist/ece_acc_{method}', ave_eces=ave_eces, ave_accs=ave_accs)
