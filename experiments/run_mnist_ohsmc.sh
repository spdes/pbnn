#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -a 0-9
#SBATCH -o ./logs/mnist_ohsmc_%a.log
#SBATCH -C "fat"
#SBATCH -t 10:00:00

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/pbnn
source ./venv/bin/activate

cd ./experiments

if [ ! -d "./results/mnist" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/mnist
fi

nvidia-smi
python mnist/ohsmc.py --id=$SLURM_ARRAY_TASK_ID --lr=0.002 --nlpd_reduce=100
