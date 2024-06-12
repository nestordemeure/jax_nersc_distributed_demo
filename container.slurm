#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=jax_distributed_container_demo
#SBATCH --account=nstaff
#SBATCH --image=ghcr.io/nvidia/jax:jax

# Run script
srun shifter --module=gpu python3 ./distributed.py
