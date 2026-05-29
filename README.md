# BERT Knowledge Distillation + Quantization

Compress large language models for deployment using Knowledge Distillation (KD) and Post-Training Quantization (PTQ) on GLUE benchmark tasks.

**Pipeline:** Fine-tuned Teacher → KD Student → INT8 Quantization

---

## Results (SST-2)

### FP32 Models

| Model | Accuracy | KD Gain | Size | Batch Latency |
|-------|----------|---------|------|---------------|
| Teacher (BERT-large) | 92.89% | — | 1278 MB | — |
| BERT-base (no KD) | 92.43% | — | 418 MB | 2911 ms |
| BERT-base + KD | 92.43% | +0.00% | 418 MB | 2899 ms |
| BERT-small (no KD) | 87.61% | — | 110 MB | 883 ms |
| **BERT-small + KD** | **89.45%** | **+1.84%** | **110 MB** | **884 ms** |
| BERT-tiny (no KD) | 82.00% | — | 17 MB | 404 ms |
| BERT-tiny + KD | 83.26% | +1.26% | 17 MB | 404 ms |

### After INT8 Quantization (PTQ)

| Model | FP32 Acc | INT8 Acc | Drop | FP32 Size | INT8 Size | Compression | Speedup |
|-------|----------|----------|------|-----------|-----------|-------------|---------|
| BERT-base (no KD) | 92.43% | 91.40% | -1.03% | 418 MB | 91 MB | 4.6x | 1.67x |
| BERT-base + KD | 92.43% | 91.74% | **-0.69%** | 418 MB | 91 MB | 4.6x | 1.66x |
| BERT-small (no KD) | 87.61% | 86.58% | -1.03% | 110 MB | 61 MB | 1.8x | 1.42x |
| BERT-small + KD | 89.45% | 88.53% | **-0.92%** | 110 MB | 61 MB | 1.8x | 1.43x |
| BERT-tiny (no KD) | 82.00% | 81.65% | -0.34% | 17 MB | 15 MB | 1.1x | 1.29x |
| **BERT-tiny + KD** | **83.26%** | **83.14%** | **-0.11%** | **17 MB** | **15 MB** | **1.1x** | **1.29x** |

### Key Findings

**Knowledge Distillation:**
- KD benefit increases as student capacity decreases: BERT-base +0% → BERT-small +1.84% → BERT-tiny +1.26%
- BERT-base shows no KD benefit due to SST-2 ceiling effect (task too easy for base-size model)

**Quantization:**
- KD-trained models are more quantization-friendly — consistently smaller accuracy drop than no-KD counterparts
- BERT-tiny + KD is nearly lossless after INT8 quantization (only **-0.11%** drop)
- BERT-base compresses 4.6x in size; BERT-tiny barely compresses (1.1x) because embedding layers dominate and are not quantized

**Full pipeline (KD + Quantization):**
- BERT-tiny + KD + INT8: **83.14%** at **15 MB** — 85x smaller than teacher, only 9.75% accuracy gap
- BERT-small + KD + INT8: **88.53%** at **61 MB** — best accuracy-efficiency tradeoff for moderate size constraints

---

## Setup

```bash
conda activate py312
pip install "optimum[onnxruntime]" ml_dtypes
```

**Requirements:** `torch`, `transformers`, `datasets`, `evaluate`, `optimum`, `onnxruntime`

---

## Usage

### 1. Train Teacher (optional)
Skip this step if using a pre-trained model from HuggingFace.

```bash
python train_teacher.py \
    --task sst2 \
    --model microsoft/deberta-v3-large \
    --output_dir ./outputs/teacher
```

### 2. Pre-compute Teacher Logits (Offline KD)
Run teacher inference once and cache soft labels to disk. This makes distillation as fast as standard fine-tuning.

```bash
python precompute_teacher.py \
    --task sst2 \
    --teacher_path assemblyai/bert-large-uncased-sst2 \
    --output_dir ./outputs/teacher_logits
```

### 3. Distill Student

**Logit-based KD** (fast):
```bash
python distill_student.py \
    --task sst2 \
    --student_model bert-base-uncased \
    --distill_mode logit \
    --teacher_logits_dir ./outputs/teacher_logits \
    --output_dir ./outputs/student
```

**Feature-based KD** (logit + attention + hidden state alignment):
```bash
python distill_student.py \
    --task sst2 \
    --teacher_path assemblyai/bert-large-uncased-sst2 \
    --student_model bert-base-uncased \
    --distill_mode feature \
    --beta 0.1 \
    --gamma 1.0 \
    --output_dir ./outputs/student_feature
```

### 4. Quantize (PTQ INT8)
```bash
python quantize.py \
    --task sst2 \
    --model_path ./outputs/student/sst2 \
    --output_dir ./outputs/quantized
```

### 5. Evaluate & Compare
```bash
python eval.py \
    --task sst2 \
    --models assemblyai/bert-large-uncased-sst2 \
             ./outputs/baseline/sst2 \
             ./outputs/student/sst2 \
    --names "Teacher (BERT-large)" \
            "BERT-base (no KD)" \
            "BERT-base + KD"
```

---

## Supported GLUE Tasks

`sst2` · `mnli` · `qqp` · `qnli` · `rte` · `mrpc` · `cola` · `stsb`

---

## Key Arguments

### `distill_student.py`
| Argument | Default | Description |
|----------|---------|-------------|
| `--distill_mode` | `logit` | `logit` or `feature` |
| `--temperature` | `4.0` | Softening temperature for KD |
| `--alpha` | `0.5` | Hard label weight (1-alpha = soft label weight) |
| `--beta` | `0.1` | Attention alignment weight (feature mode) |
| `--gamma` | `1.0` | Hidden state alignment weight (feature mode) |
| `--teacher_logits_dir` | `None` | Path to pre-computed logits (offline KD) |
| `--max_steps` | `-1` | Limit steps for quick smoke test |

---

## Project Structure

```
distill_quant/
├── utils.py                # GLUE task configs, tokenization, metrics
├── train_teacher.py        # Fine-tune teacher model
├── precompute_teacher.py   # Cache teacher soft labels (offline KD)
├── distill_student.py      # Knowledge distillation (logit & feature)
├── quantize.py             # Post-training INT8 quantization
├── eval.py                 # Compare models: accuracy / size / latency
└── slurm/                  # SLURM job scripts
```

---

## Distillation Loss

**Logit KD:**

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(y, \hat{y}) + (1 - \alpha) \cdot T^2 \cdot D_{KL}\left( \frac{z_S}{T} \,\middle\|\middle\|\, \frac{z_T}{T} \right)
$$

**Feature KD** (additional terms):

$$
\mathcal{L}_{feat} = \beta \cdot \mathcal{L}_{attn} + \gamma \cdot \mathcal{L}_{hidden}
$$

where attention maps are averaged over heads to handle head-count mismatch, and hidden states are aligned via a learnable linear projection.
