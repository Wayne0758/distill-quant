#!/bin/bash
#SBATCH --job-name=distill_student
#SBATCH --output=../outputs/student_%j.log
#SBATCH --error=../outputs/student_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/miniconda3/bin/activate py312
cd ~/distill_quant

python distill_student.py \
    --task sst2 \
    --teacher_path ./outputs/teacher/sst2 \
    --student_model bert-base-uncased \
    --output_dir ./outputs/student \
    --batch_size 32 \
    --lr 5e-5 \
    --num_epochs 5 \
    --temperature 4.0 \
    --alpha 0.5
