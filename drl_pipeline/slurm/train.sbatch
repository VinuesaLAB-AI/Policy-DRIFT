#!/bin/bash
# Train one controller: 1 agent process + 64 training DNS ranks + 64 validation DNS ranks.
#   sbatch -A <account> -p <partition> slurm/train.sbatch --reward trg
# Submit from drl_pipeline/ with the Python environment active; the arguments go to train.py.
#SBATCH --job-name=policy-drift-train
#SBATCH --nodes=1
#SBATCH --ntasks=129
#SBATCH --time=14-00:00:00
#SBATCH --output=outputs/slurm-%j.out

source slurm/mpi_env.sh
python train.py "$@"
