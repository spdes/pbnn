#!/bin/bash
#SBATCH -A snic2022-22-1110
#SBATCH -a 0-99
#SBATCH -o ./logs/moon_cpu_%a.log
#SBATCH -p main
#SBATCH -n 20
#SBATCH --mem=64G
#SBATCH --time=03:00:00

source ~/.bashrc

cd $WRKDIR/pbnn
source ./venv/bin/activate

cd ./experiments

if [ ! -d "./results/moon" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/moon
fi

python moon/map_hmc.py --id=$SLURM_ARRAY_TASK_ID
python moon/ohsmc.py --id=$SLURM_ARRAY_TASK_ID
python moon/sgsmc_hmc.py --id=$SLURM_ARRAY_TASK_ID
python moon/swag.py --id=$SLURM_ARRAY_TASK_ID --adam
python moon/vb.py --id=$SLURM_ARRAY_TASK_ID
