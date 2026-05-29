import argparse
import os
import time
import copy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from utils import TASK_CONFIG, get_dataset, get_tokenize_fn, get_val_split, get_compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",        default="sst2",                   type=str)
    parser.add_argument("--model_path",  default="./outputs/student/sst2", type=str)
    parser.add_argument("--output_dir",  default="./outputs/quantized",    type=str)
    parser.add_argument("--max_length",  default=128,                      type=int)
    parser.add_argument("--batch_size",  default=64,                       type=int)
    return parser.parse_args()


def get_model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1024 / 1024


def evaluate_model(model, tokenizer, val_data, task, max_length, batch_size):
    compute_metrics = get_compute_metrics(task)
    all_preds, all_labels = [], []
    latencies = []

    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i: i + batch_size]
            key1, key2 = TASK_CONFIG[task]["keys"]
            if key2 is None:
                texts = batch[key1]
            else:
                texts = list(zip(batch[key1], batch[key2]))
            inputs = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            t0 = time.perf_counter()
            outputs = model(**inputs)
            latencies.append(time.perf_counter() - t0)
            all_preds.append(outputs.logits.cpu().numpy())
            all_labels.extend(batch["label"])

    all_preds = np.concatenate(all_preds, axis=0)
    metrics = compute_metrics((all_preds, np.array(all_labels)))
    return metrics, np.mean(latencies) * 1000


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    fp32_model.eval()

    dataset = get_dataset(args.task)
    val_data = dataset[get_val_split(args.task)]

    # Apply PyTorch dynamic INT8 quantization (quantizes all nn.Linear layers)
    print("Applying INT8 dynamic quantization...")
    int8_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(fp32_model),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Size comparison
    fp32_size = get_model_size_mb(fp32_model)
    int8_size  = get_model_size_mb(int8_model)
    print(f"Model size  FP32: {fp32_size:.1f} MB  →  INT8: {int8_size:.1f} MB  ({int8_size/fp32_size*100:.1f}%)")

    # Accuracy + latency
    print("Evaluating FP32 model...")
    fp32_metrics, fp32_lat = evaluate_model(fp32_model, tokenizer, val_data, args.task, args.max_length, args.batch_size)
    print("Evaluating INT8 model...")
    int8_metrics, int8_lat = evaluate_model(int8_model, tokenizer, val_data, args.task, args.max_length, args.batch_size)

    metric_key = list(fp32_metrics.keys())[0]
    fp32_acc = fp32_metrics[metric_key] * 100
    int8_acc = int8_metrics[metric_key] * 100

    print(f"\n{'':30} {'FP32':>10}  {'INT8':>10}  {'Change':>10}")
    print("-" * 65)
    print(f"{'Accuracy (' + metric_key + ')':30} {fp32_acc:>9.2f}%  {int8_acc:>9.2f}%  {int8_acc - fp32_acc:>+9.2f}%")
    print(f"{'Size (MB)':30} {fp32_size:>10.1f}  {int8_size:>10.1f}  {int8_size/fp32_size:>9.1f}x")
    print(f"{'Latency (ms/batch)':30} {fp32_lat:>10.2f}  {int8_lat:>10.2f}  {fp32_lat/int8_lat:>9.2f}x")

    # Save quantized model (state dict, since dynamic quant isn't save_pretrained compatible)
    save_path = os.path.join(output_dir, "int8_model.pt")
    torch.save(int8_model.state_dict(), save_path)
    # Save tokenizer and config for later use
    tokenizer.save_pretrained(output_dir)
    fp32_model.config.save_pretrained(output_dir)
    print(f"\nQuantized model saved to {output_dir}")


if __name__ == "__main__":
    main()
