import datasets
import jax
import numpy as np
from typing import Any
from datasets import load_dataset

Array = Any
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

        
def get_glue_data(task_name):
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset("glue", task_name)
    
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    return raw_datasets, num_labels
    
    
def text_to_tokens(raw_datasets, task_name, tokenizer, max_seq_length=124):
    def preprocess_function(examples):
        sentence1_key, sentence2_key = task_to_keys[task_name]

        texts = (
            (examples[sentence1_key],) if sentence2_key is None 
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length", max_length=max_seq_length, truncation=True)

        if "label" in examples:
            result["labels"] = examples["label"]
        return result
    
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]
    return train_dataset, eval_dataset


def tokens_to_embeddings(tok_dataset, model, batch_size):
    batch_gen = glue_eval_data_collator(tok_dataset, batch_size)
    results = []
    labels = []
    for batch in batch_gen:
        labels.append(batch.pop("labels"))
        results.append(np.array(model(**batch).last_hidden_state))
    return {'embeddings': np.concatenate(results, axis=0), 'labels': np.concatenate(labels)}
    
    
def glue_train_data_collator(rng: PRNGKey, dataset: Dataset, batch_size: int):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`"""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: np.array(v) for k, v in batch.items()}
        yield batch
        
        
        
def glue_eval_data_collator(dataset: Dataset, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`. """
    batch_idx = np.arange(len(dataset))

    steps_per_epoch = np.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch