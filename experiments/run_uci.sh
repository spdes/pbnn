#!/bin/bash

sbatch ./run_uci_batch.sh boston 50
sbatch ./run_uci_batch.sh concrete 50
sbatch ./run_uci_batch.sh energy 50
sbatch ./run_uci_batch.sh kin8 50
sbatch ./run_uci_batch.sh naval 50
sbatch ./run_uci_batch.sh yacht 20
sbatch ./run_uci_batch.sh protein 100
sbatch ./run_uci_batch.sh power 50

sbatch ./run_uci_batch.sh australian 50
sbatch ./run_uci_batch.sh cancer 50
sbatch ./run_uci_batch.sh ionosphere 20
sbatch ./run_uci_batch.sh glass 20
sbatch ./run_uci_batch.sh satellite 50
