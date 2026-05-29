import argparse
import os
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import DataLoader
from utils import TASK_CONFIG, get_dataset, get_tokenize_fn, get_compute_metrics, get_val_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       default="sst2",  type=str)
    parser.add_argument("--models",     nargs="+",       required=True,
                        help="List of model paths or HF IDs to compare")
    parser.add_argument("--names",      nargs="+",       default=None,
                        help="Display names for each model (same order as --models)")
    parser.add_argument("--max_length", default=128,     type=int)
    parser.add_argument("--batch_size", default=64,      type=int)
    return parser.parse_args()


def get_model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1024 / 1024


def evaluate_model(model_path, task, max_length, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    dataset = get_dataset(task)
    val_data = dataset[get_val_split(task)]
    tokenize_fn = get_tokenize_fn(tokenizer, task, max_length)
    val_data = val_data.map(tokenize_fn, batched=True)
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label",
                                                "token_type_ids"] if "token_type_ids" in val_data.column_names else ["input_ids", "attention_mask", "label"])

    compute_metrics = get_compute_metrics(task)
    is_regression = (task == "stsb")

    all_preds, all_labels = [], []
    latencies = []

    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i: i + batch_size]
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"]

            t0 = time.perf_counter()
            outputs = model(**inputs)
            latencies.append(time.perf_counter() - t0)

            logits = outputs.logits.cpu().numpy()
            all_preds.append(logits)
            all_labels.extend(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.array(all_labels)
    metrics = compute_metrics((all_preds, all_labels))

    return {
        "metrics": metrics,
        "size_mb": get_model_size_mb(model),
        "avg_latency_ms": np.mean(latencies) * 1000,
        "num_params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }


def main():
    args = parse_args()
    names = args.names if args.names else args.models

    results = {}
    for model_path, name in zip(args.models, names):
        print(f"\nEvaluating: {name} ...")
        results[name] = evaluate_model(model_path, args.task, args.max_length, args.batch_size)

    # Print comparison table
    print("\n" + "=" * 75)
    print(f"  Task: {args.task.upper()}")
    print("=" * 75)

    # Collect metric keys
    metric_keys = list(next(iter(results.values()))["metrics"].keys())

    header = f"{'Model':<30} " + " ".join(f"{k:>10}" for k in metric_keys) + f"  {'Size(MB)':>10}  {'Params(M)':>10}  {'Lat(ms)':>8}"
    print(header)
    print("-" * 75)

    for name, r in results.items():
        metric_vals = " ".join(f"{v*100:>9.2f}%" for v in r["metrics"].values())
        print(f"{name:<30} {metric_vals}  {r['size_mb']:>10.1f}  {r['num_params_m']:>10.1f}  {r['avg_latency_ms']:>8.1f}")

    print("=" * 75)


if __name__ == "__main__":
    main()
