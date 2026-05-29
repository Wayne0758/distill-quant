#!/bin/bash
#SBATCH --job-name=distill_teacher
#SBATCH --output=../outputs/teacher_%j.log
#SBATCH --error=../outputs/teacher_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/miniconda3/bin/activate py312
cd ~/distill_quant

python train_teacher.py \
    --task sst2 \
    --model microsoft/deberta-v3-large \
    --output_dir ./outputs/teacher \
    --batch_size 16 \
    --lr 2e-5 \
    --num_epochs 5
