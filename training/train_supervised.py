
import json
import os
import random
from glob import glob

import evaluate
import fire
import numpy as np
import pandas as pd
from datasets import Dataset
from termcolor import colored
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

def tokenize_fn(
    examples,
    tokenizer: AutoTokenizer,
) -> dict:
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def load_datafile(
    dataset_file: str,
    machine_key: str,
    tokenizer: AutoTokenizer,
    num_datapoints: int = None,
    debug: bool = False,
) -> Dataset:
    nrows = 3 * 100 if debug else None
    df = pd.read_json(dataset_file, lines=True, nrows=nrows)
    if "author_id" not in df:
        df["author_id"] = [str(x) for x in range(len(df))]
    df = df[["author_id", "content_text", machine_key]]
    df = df.groupby("author_id").agg(list)

    records = []
    
    def fix(record: dict) -> dict:
        if not isinstance(record["text"], list):
            return record
        if isinstance(record["text"][0], list):
            record["text"] = record["text"][0][0]
        else:
            record["text"] = record["text"][0]
        return record

    for _, row in df.iterrows():
        randindex = random.randint(0, len(row["content_text"])-1)
        # sampling in this way ensures that human / machine are 
        # from the same topic:
        records.append(fix({
            "text": row["content_text"][randindex],
            "label": 0,
        }))
        records.append(fix({
            "text": row[machine_key][randindex],
            "label": 1,
        }))

        if num_datapoints is not None and len(records) >= num_datapoints:
            break
    
    dataset = Dataset.from_list(records)
    
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=40,
        fn_kwargs={"tokenizer": tokenizer},
    )
    return dataset

def get_filename(
    filenames: list[str], 
    split: str,
    filename_substring: str,
):
    if filename_substring:
        valid_names = [fname for fname in filenames if filename_substring in fname and split in fname]
    else:
        valid_names = [fname for fname in filenames if split in fname]

    assert len(valid_names) == 1
    return valid_names[0]

def load_data(
    dataset_path: str,
    machine_key: str,
    tokenizer: AutoTokenizer,
    num_train_datapoints: int,
    filename_substring: str = None,
    debug: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    filenames = glob(os.path.join(dataset_path, "*.jsonl*"))
    train_filename = get_filename(filenames, "train", filename_substring)
    valid_filename = get_filename(filenames, "valid", filename_substring)
    test_filename = get_filename(filenames, "test", filename_substring)

    train_dataset = load_datafile(
        train_filename,
        machine_key,
        tokenizer, 
        num_datapoints=num_train_datapoints, 
        debug=debug
    )

    valid_dataset = load_datafile(
        valid_filename, 
        machine_key,
        tokenizer, 
        debug=debug
    )

    test_dataset = load_datafile(
        test_filename, 
        machine_key,
        tokenizer, 
        debug=debug
    )

    return (train_dataset, valid_dataset, test_dataset)

def main(
    dataset_path: str = "./data/MTD_reddit_politics_1000_Mistral-7B-Instruct-v0.3_N=5_transfer_N=10_temp=0.7_top-p=0.9_model=MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=False_8-3",
    machine_key: str = "paraphrase_content_text",
    num_train_datapoints: int = 1_000,
    model_name: str = "roberta-base",
    per_device_train_batch_size = 128,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    num_train_epochs = 10.0,
    filename_substring: str = None,
    debug: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, valid_dataset, test_dataset = \
        load_data(dataset_path, machine_key, tokenizer, num_train_datapoints, filename_substring=filename_substring, debug=debug)

    metric = evaluate.load("accuracy")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    output_dir = os.path.join(dataset_path, "checkpoints")
    run_name = f"{model_name}_{machine_key}-{num_train_datapoints}"
    run_name += "-debug" if debug else ""
    output_dir = os.path.join(output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(colored(f"output_dir: {output_dir}", "yellow"))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    hparams = {
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
    }
    with open(os.path.join(output_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f)

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=43,
        bf16=True,
        load_best_model_at_end=True,
        optim="adamw_torch",
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        **hparams,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "best"))

    output = trainer.evaluate(test_dataset)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=4)

    return 0

if __name__ == "__main__":
    set_seed(43)
    fire.Fire(main)