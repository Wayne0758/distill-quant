import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from utils import TASK_CONFIG, get_dataset, get_tokenize_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",         default="sst2",                              type=str)
    parser.add_argument("--teacher_path", default="assemblyai/bert-large-uncased-sst2", type=str)
    parser.add_argument("--output_dir",   default="./outputs/teacher_logits",          type=str)
    parser.add_argument("--max_length",   default=128,                                 type=int)
    parser.add_argument("--batch_size",   default=128,                                 type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher_path).to(device)
    model.eval()

    dataset = get_dataset(args.task)
    tokenize_fn = get_tokenize_fn(tokenizer, args.task, args.max_length)

    for split in ["train", "validation"]:
        actual_split = "validation_matched" if args.task == "mnli" and split == "validation" else split
        data = dataset[actual_split].map(tokenize_fn, batched=True)
        data.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"]
                        if "token_type_ids" in data.column_names else ["input_ids", "attention_mask"])

        loader = DataLoader(data, batch_size=args.batch_size)
        all_logits = []

        print(f"Computing teacher logits for {split} ({len(data)} examples)...")
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                logits = model(**inputs).logits.cpu().numpy()
                all_logits.append(logits)

        all_logits = np.concatenate(all_logits, axis=0)
        save_path = os.path.join(args.output_dir, args.task, f"{split}_logits.npy")
        np.save(save_path, all_logits)
        print(f"Saved {all_logits.shape} logits → {save_path}")


if __name__ == "__main__":
    main()
