import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from utils import TASK_CONFIG, get_dataset, get_tokenize_fn, get_compute_metrics, get_val_split


class GlueWithLogits(Dataset):
    """Wraps a HuggingFace dataset and attaches pre-computed teacher logits."""
    def __init__(self, hf_dataset, logits):
        self.dataset = hf_dataset
        self.logits  = torch.tensor(logits, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        TENSOR_COLS = {"input_ids", "attention_mask", "token_type_ids", "label"}
        item = {k: torch.tensor(v) for k, v in self.dataset[idx].items()
                if k in TENSOR_COLS}
        item["teacher_logits"] = self.logits[idx]
        return item


# ---------------------------------------------------------------------------
# Feature-mode: wrap student with learnable projections (hidden: s→t size)
# ---------------------------------------------------------------------------
class StudentWithProjections(nn.Module):
    def __init__(self, student_model, teacher_hidden_size):
        super().__init__()
        self.base = student_model
        self.config = student_model.config
        s_hidden = student_model.config.hidden_size
        num_layers = student_model.config.num_hidden_layers
        # one projection per layer (embedding + each transformer layer)
        self.projections = nn.ModuleList([
            nn.Linear(s_hidden, teacher_hidden_size, bias=False)
            for _ in range(num_layers + 1)
        ])

    def forward(self, **kwargs):
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("output_attentions", True)
        return self.base(**kwargs)

    def save_pretrained(self, path, **kwargs):
        self.base.save_pretrained(path, **kwargs)


# ---------------------------------------------------------------------------
# Logit-only KD  (soft label + hard label)
# ---------------------------------------------------------------------------
class LogitDistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature, alpha, is_regression=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model if teacher_model is not None else None
        if self.teacher is not None:
            self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.is_regression = is_regression

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # offline: teacher_logits 已預存在 batch 裡
        if "teacher_logits" in inputs:
            teacher_logits = inputs.pop("teacher_logits")
        else:
            with torch.no_grad():
                teacher_logits = self.teacher(**inputs).logits

        outputs = model(**inputs)
        student_logits = outputs.logits

        if self.is_regression:
            loss_hard = F.mse_loss(student_logits.squeeze(), labels.float())
            loss_soft = F.mse_loss(student_logits.squeeze(), teacher_logits.squeeze())
        else:
            T = self.temperature
            loss_hard = F.cross_entropy(student_logits, labels)
            loss_soft = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
            ) * T * T

        loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Feature-based KD  (logit + attention + hidden state)
# ---------------------------------------------------------------------------
class FeatureDistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature, alpha, beta, gamma,
                 is_regression=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha   # logit KD weight
        self.beta  = beta    # attention alignment weight
        self.gamma = gamma   # hidden state alignment weight
        self.is_regression = is_regression

    @staticmethod
    def _layer_map(num_student, num_teacher):
        # uniform stride: student i → teacher (i+1)*stride - 1
        stride = num_teacher // num_student
        return {i: (i + 1) * stride - 1 for i in range(num_student)}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # student forward (model is StudentWithProjections)
        outputs = model(**inputs)
        s_logits  = outputs.logits
        s_hiddens = outputs.hidden_states  # tuple: (num_layers+1) × (B, L, s_hidden)
        s_attns   = outputs.attentions     # tuple: num_layers × (B, heads_s, L, L)

        with torch.no_grad():
            t_out     = self.teacher(**inputs, output_hidden_states=True, output_attentions=True)
            t_logits  = t_out.logits
            t_hiddens = t_out.hidden_states
            t_attns   = t_out.attentions

        lmap = self._layer_map(len(s_attns), len(t_attns))

        # 1. Logit loss
        if self.is_regression:
            loss_hard = F.mse_loss(s_logits.squeeze(), labels.float())
            loss_soft = F.mse_loss(s_logits.squeeze(), t_logits.squeeze())
        else:
            T = self.temperature
            loss_hard = F.cross_entropy(s_logits, labels)
            loss_soft = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * T * T
        loss_logit = self.alpha * loss_hard + (1 - self.alpha) * loss_soft

        # 2. Attention loss (average over heads to handle head-count mismatch)
        loss_attn = sum(
            F.mse_loss(s_attns[si].mean(1), t_attns[ti].mean(1))
            for si, ti in lmap.items()
        ) / len(lmap)

        # 3. Hidden state loss (project student hidden → teacher hidden size)
        loss_hidden = sum(
            F.mse_loss(model.projections[si](s_hiddens[si + 1]),
                       t_hiddens[ti + 1].detach())
            for si, ti in lmap.items()
        ) / len(lmap)

        loss = loss_logit + self.beta * loss_attn + self.gamma * loss_hidden
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Args & main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",          default="sst2",                              type=str)
    parser.add_argument("--teacher_path",  default="textattack/bert-large-uncased-SST-2", type=str)
    parser.add_argument("--student_model", default="bert-base-uncased",                 type=str)
    parser.add_argument("--output_dir",    default="./outputs/student",                 type=str)
    parser.add_argument("--max_length",    default=128,                                 type=int)
    parser.add_argument("--batch_size",    default=32,                                  type=int)
    parser.add_argument("--lr",            default=5e-5,                                type=float)
    parser.add_argument("--num_epochs",    default=5,                                   type=int)
    parser.add_argument("--temperature",   default=4.0,                                 type=float)
    parser.add_argument("--alpha",         default=0.5,                                 type=float,
                        help="Hard label weight (1-alpha = soft label weight)")
    parser.add_argument("--distill_mode",  default="logit", choices=["logit", "feature"], type=str)
    parser.add_argument("--beta",          default=0.1,                                 type=float,
                        help="[feature] Attention alignment weight")
    parser.add_argument("--gamma",         default=1.0,                                 type=float,
                        help="[feature] Hidden state alignment weight")
    parser.add_argument("--seed",              default=42,    type=int)
    parser.add_argument("--max_steps",         default=-1,    type=int,
                        help="Set >0 to limit steps for quick smoke test")
    parser.add_argument("--teacher_logits_dir", default=None, type=str,
                        help="Path to pre-computed teacher logits (offline KD). "
                             "If set, teacher_path is not needed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = os.path.join(args.output_dir, args.task)
    num_labels = TASK_CONFIG[args.task]["num_labels"]
    is_regression = (args.task == "stsb")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Teacher (frozen) — skip if using offline logits
    offline = args.teacher_logits_dir is not None
    if not offline:
        teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_path)
        teacher_model = teacher_model.to(device)
        for p in teacher_model.parameters():
            p.requires_grad = False
    else:
        teacher_model = None
        print("Offline KD: loading pre-computed teacher logits.")

    # Student
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    base_student = AutoModelForSequenceClassification.from_pretrained(
        args.student_model, num_labels=num_labels
    )

    dataset = get_dataset(args.task)
    dataset = dataset.map(get_tokenize_fn(student_tokenizer, args.task, args.max_length), batched=True)

    # Attach pre-computed teacher logits if offline mode
    if offline:
        logits_dir = os.path.join(args.teacher_logits_dir, args.task)
        train_logits = np.load(os.path.join(logits_dir, "train_logits.npy"))
        val_logits   = np.load(os.path.join(logits_dir, "validation_logits.npy"))
        train_dataset = GlueWithLogits(dataset["train"], train_logits)
        val_dataset   = GlueWithLogits(dataset[get_val_split(args.task)], val_logits)
    else:
        train_dataset = dataset["train"]
        val_dataset   = dataset[get_val_split(args.task)]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=args.seed,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    common = dict(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=student_tokenizer,
        compute_metrics=get_compute_metrics(args.task),
        data_collator=default_data_collator if offline else None,
    )

    if args.distill_mode == "feature":
        teacher_hidden = teacher_model.config.hidden_size
        student_model = StudentWithProjections(base_student, teacher_hidden)
        trainer = FeatureDistillationTrainer(
            teacher_model=teacher_model,
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            is_regression=is_regression,
            model=student_model,
            **common,
        )
    else:
        trainer = LogitDistillationTrainer(
            teacher_model=teacher_model,
            temperature=args.temperature,
            alpha=args.alpha,
            is_regression=is_regression,
            model=base_student,
            **common,
        )

    trainer.train()
    trainer.save_model(output_dir)
    student_tokenizer.save_pretrained(output_dir)
    print(f"[{args.distill_mode}] Student saved to {output_dir}")


if __name__ == "__main__":
    main()
