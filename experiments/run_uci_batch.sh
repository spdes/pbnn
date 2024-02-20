#!/bin/bash
#SBATCH -A Berzelius-2023-194
#SBATCH --gpus=1
#SBATCH -a 0-9
#SBATCH -o ./logs/uci_%a.log
#SBATCH -C "thin"
#SBATCH -t 10:00:00

data_name=$1
batch_size=$2

source ~/.bashrc
export XLA_PYTHON_CLIENT_PREALLOCATE=true

cd $WRKDIR/pbnn
source ./venv/bin/activate

cd ./experiments

if [ ! -d "./results/uci" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/uci
fi

python uci/map_hmc.py --id=$SLURM_ARRAY_TASK_ID --data_name=$data_name --batch_size=$batch_size
python uci/ohsmc.py --id=$SLURM_ARRAY_TASK_ID --data_name=$data_name --batch_size=$batch_size
python uci/ohsmc_ou.py --id=$SLURM_ARRAY_TASK_ID --data_name=$data_name --batch_size=$batch_size
python uci/swag.py --id=$SLURM_ARRAY_TASK_ID --adam --data_name=$data_name --batch_size=$batch_size
python uci/vb.py --id=$SLURM_ARRAY_TASK_ID --data_name=$data_name --batch_size=$batch_size
python uci/sgsmc_hmc.py --id=$SLURM_ARRAY_TASK_ID --data_name=$data_name --batch_size=$batch_size