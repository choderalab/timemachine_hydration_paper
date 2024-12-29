#!/bin/bash

#SBATCH --partition gpu
#SBATCH --ntasks=1
#SBATCH --array=1-643
#SBATCH --time=8:00:00
#SBATCH --job-name=tm-freesolv-esp
#SBATCH --output=%A_%a_%x_%N.out
#SBATCH --error=%A_%a_%x_%N.err
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus-per-task=1

# Source bashrc (useful for working conda/mamba)
source ${HOME}/.bashrc

# Activate environment
conda activate tm_off

# Report node in use
hostname

# Report CUDA info
env | sort | grep 'CUDA'

# set loglevel debug
LOGLEVEL=DEBUG

mkdir /scratch/choderaj/$USER/tm-freesolv-esp_$(( ${SLURM_ARRAY_TASK_ID} - 1 ))
export TMPDIR=/scratch/choderaj/$USER/tm-freesolv-esp_$(( ${SLURM_ARRAY_TASK_ID} - 1 ))

# Check if TMPDIR is set so we don't do rm -rf /* by mistake
if [ -z "$TMPDIR" ]; then
    echo "Error: TMPDIR is not set."
    exit 1
fi
clean_tmpdir() {
    echo "Cleaning up $TMPDIR"
    rm -rf $TMPDIR/*
    # You could also put stuff in there like a cp command to move a file off of local scratch
}

# Register the cleanup function to be called upon job completion or termination
trap 'clean_tmpdir' EXIT

# run this...
python run_freesolv.opt.py --idx $(( ${SLURM_ARRAY_TASK_ID} - 1 )) --seed 2024
