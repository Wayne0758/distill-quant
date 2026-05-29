import numpy as np
import evaluate
from datasets import load_dataset

TASK_CONFIG = {
    "sst2": {"keys": ("sentence",  None),           "num_labels": 2},
    "cola": {"keys": ("sentence",  None),           "num_labels": 2},
    "mrpc": {"keys": ("sentence1", "sentence2"),    "num_labels": 2},
    "rte":  {"keys": ("sentence1", "sentence2"),    "num_labels": 2},
    "qqp":  {"keys": ("question1", "question2"),    "num_labels": 2},
    "qnli": {"keys": ("question",  "sentence"),     "num_labels": 2},
    "mnli": {"keys": ("premise",   "hypothesis"),   "num_labels": 3},
    "stsb": {"keys": ("sentence1", "sentence2"),    "num_labels": 1},
}

def get_dataset(task):
    return load_dataset("glue", task)

def get_tokenize_fn(tokenizer, task, max_length=128):
    key1, key2 = TASK_CONFIG[task]["keys"]
    def tokenize(examples):
        args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        return tokenizer(*args, truncation=True, max_length=max_length, padding="max_length")
    return tokenize

def get_compute_metrics(task):
    metric = evaluate.load("glue", task)
    is_regression = (task == "stsb")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.squeeze() if is_regression else np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)
    return compute_metrics

def get_val_split(task):
    return "validation_matched" if task == "mnli" else "validation"
