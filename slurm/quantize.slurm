#!/bin/bash
#SBATCH --job-name=distill_quantize
#SBATCH --output=../outputs/quantize_%j.log
#SBATCH --error=../outputs/quantize_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/miniconda3/bin/activate py312
cd ~/distill_quant

python quantize.py \
    --task sst2 \
    --model_path ./outputs/student/sst2 \
    --output_dir ./outputs/quantized
