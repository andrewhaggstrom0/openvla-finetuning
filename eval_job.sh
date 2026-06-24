#!/bin/bash
#SBATCH -p condo-cse5100
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --mem=40G
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -J openvla_eval
#SBATCH -A engr-acad-cse5100
#SBATCH -o /home/compute/a.haggstrom/openvla_project/logs/slurm_eval_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.haggstrom@wustl.edu

echo "Job started on $(hostname) at $(date)"
nvidia-smi

export HF_HOME="/home/compute/a.haggstrom/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/compute/a.haggstrom/.local/lib/python3.9/site-packages:$PYTHONPATH"

find ~/.cache/huggingface/modules -name "*.pyc" -delete 2>/dev/null
find ~/.cache/huggingface/modules -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

cd /home/compute/a.haggstrom/openvla_project
python3 evaluate.py

echo "Job finished at $(date)"
