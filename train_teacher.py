import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from utils import TASK_CONFIG, get_dataset, get_tokenize_fn, get_compute_metrics, get_val_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",          default="sst2",                         type=str)
    parser.add_argument("--model",         default="microsoft/deberta-v3-large",   type=str)
    parser.add_argument("--output_dir",    default="./outputs/teacher",            type=str)
    parser.add_argument("--max_length",    default=128,                            type=int)
    parser.add_argument("--batch_size",    default=16,                             type=int)
    parser.add_argument("--lr",            default=2e-5,                           type=float)
    parser.add_argument("--num_epochs",    default=5,                              type=int)
    parser.add_argument("--seed",          default=42,                             type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = os.path.join(args.output_dir, args.task)
    num_labels = TASK_CONFIG[args.task]["num_labels"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    dataset = get_dataset(args.task)
    dataset = dataset.map(get_tokenize_fn(tokenizer, args.task, args.max_length), batched=True)

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
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[get_val_split(args.task)],
        tokenizer=tokenizer,
        compute_metrics=get_compute_metrics(args.task),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Teacher saved to {output_dir}")


if __name__ == "__main__":
    main()
