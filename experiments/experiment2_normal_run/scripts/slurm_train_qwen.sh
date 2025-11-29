#!/bin/bash
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=baraa.abed@duke.edu
#SBATCH --mem=32G
#SBATCH --output=/hpc/home/bfa6/work/github/yapper/experiments/experiment2_normal_run/slurm_logs/%x_%j.out
#SBATCH --error=/hpc/home/bfa6/work/github/yapper/experiments/experiment2_normal_run/slurm_logs/%x_%j.err

export PYTHONUNBUFFERED=1

source /hpc/home/bfa6/work/github/yapper/.venv/bin/activate

python /hpc/home/bfa6/work/github/yapper/experiments/experiment2_normal_run/scripts/train_qwen.py