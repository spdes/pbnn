#!/bin/bash
#SBATCH -A snic2022-22-1110
#SBATCH -a 0-99
#SBATCH -o ./logs/regression_cpu_%a.log
#SBATCH -p main
#SBATCH -n 20
#SBATCH --mem=64G
#SBATCH --time=03:00:00

source ~/.bashrc

cd $WRKDIR/pbnn
source ./venv/bin/activate

cd ./experiments

if [ ! -d "./results/regression" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/regression
fi

python regression/map_hmc.py --id=$SLURM_ARRAY_TASK_ID
python regression/ohsmc.py --id=$SLURM_ARRAY_TASK_ID
python regression/sgsmc_hmc.py --id=$SLURM_ARRAY_TASK_ID
python regression/swag.py --id=$SLURM_ARRAY_TASK_ID --adam
python regression/vb.py --id=$SLURM_ARRAY_TASK_ID
