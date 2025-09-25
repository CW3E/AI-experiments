#!/bin/bash
#SBATCH --account=cwp179
#SBATCH --partition=cw3e-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-8
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --job-name=GCinfer
#SBATCH --output=./logs/GCinfer_%j.log
#SBATCH --error=./logs/GCinfer_%j.error

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python graphcast_inference.py
# python graphcast_inference_clim.py
