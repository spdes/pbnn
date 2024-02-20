#!/bin/bash
#SBATCH -A Berzelius-2023-146
#SBATCH --gpus=1
#SBATCH -o ./logs/mnist_ece_acc.log
#SBATCH -C "thin"
#SBATCH -t 01:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/pbnn
source ./venv_torch/bin/activate

cd ./experiments

if [ ! -d "./results/mnist" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/mnist
fi

nvidia-smi
python tabulators/tabulate_mnist.py
