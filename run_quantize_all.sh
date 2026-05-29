#!/bin/bash
# Quantize all 6 trained models (3 sizes x 2: no-KD and KD)
set -e
cd /home/wayne0758/distill_quant

declare -A MODELS
MODELS=(
    ["BERT-base (no KD)"]="baseline/sst2        quantized/baseline"
    ["BERT-base + KD"]="student/sst2         quantized/student_base"
    ["BERT-small (no KD)"]="baseline_small/sst2  quantized/baseline_small"
    ["BERT-small + KD"]="student_small/sst2   quantized/student_small"
    ["BERT-tiny (no KD)"]="baseline_tiny/sst2   quantized/baseline_tiny"
    ["BERT-tiny + KD"]="student_tiny/sst2    quantized/student_tiny"
)

ORDER=(
    "BERT-base (no KD)"
    "BERT-base + KD"
    "BERT-small (no KD)"
    "BERT-small + KD"
    "BERT-tiny (no KD)"
    "BERT-tiny + KD"
)

for name in "${ORDER[@]}"; do
    paths="${MODELS[$name]}"
    model_path=$(echo $paths | awk '{print $1}')
    output_dir=$(echo $paths | awk '{print $2}')
    echo ""
    echo "========================================"
    echo "  Quantizing: $name"
    echo "========================================"
    python quantize.py \
        --task sst2 \
        --model_path ./outputs/$model_path \
        --output_dir ./outputs/$output_dir
done

echo ""
echo "All done. Results logged above."
