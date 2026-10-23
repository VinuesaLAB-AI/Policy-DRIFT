#!/bin/bash
# Evaluate one policy on the six held-out initial conditions: 1 agent process + 64 DNS ranks.
#   sbatch -A <account> -p <partition> slurm/evaluate.sbatch --model ../checkpoints/drl/policy_drift_td3.zip
# Submit from drl_pipeline/ with the Python environment active; the arguments go to evaluate.py.
#SBATCH --job-name=policy-drift-eval
#SBATCH --nodes=1
#SBATCH --ntasks=65
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm-%j.out

source slurm/mpi_env.sh
python evaluate.py "$@"
