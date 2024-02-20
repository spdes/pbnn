#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -o ./logs/mnist_time.log
#SBATCH -C "thin"
#SBATCH -t 00:30:00

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
python time_profile/ohsmc.py --batch_size=20 --nsamples=100
python time_profile/vb.py --batch_size=20 --vbsamples=100
python time_profile/swag.py --batch_size=20
