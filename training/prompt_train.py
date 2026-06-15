
import os
import sys
from  math import ceil
from argparse import ArgumentParser
from collections import OrderedDict

import ujson as json
import torch
from accelerate import PartialState
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from termcolor import colored
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    set_seed,
)
from trl import (
    SFTConfig, 
    SFTTrainer, 
    DataCollatorForCompletionOnlyLM,
)

torch.autograd.set_detect_anomaly(True)

set_seed(43)
# Where SFT checkpoints get written. Override via STYLE_AWARE_OUTPUT.
OUTPUT_DIR = os.environ.get(
    "STYLE_AWARE_OUTPUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"),
)

parser = ArgumentParser()
parser.add_argument("--model_name", type=str,
                    default="mistralai/Mistral-7B-v0.3",
                    help="Name of the model to run.")
parser.add_argument("--dataset_path", type=str,
                    default="./data/HF/MUD_paraphrase_Mistral-7B-Instruct-v0.3_N=5_small=True_8-3",
                    help="Directory where the dataset is stored.")
parser.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate.")
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=999_999, # 2500 for batch_size of 128 on small=True
                    help="Number of training steps.")
parser.add_argument("--max_seq_length", type=int, default=None)
parser.add_argument("--resume_from_checkpoint", default=False,
                    action="store_true",
                    help="Whether to resume from a checkpoint.")
parser.add_argument("--save_steps", type=int, default=200,
                    help="Number of steps to save the model.")
parser.add_argument("--logging_steps", type=int, default=100,
                    help="Number of steps to log the training metrics.")
parser.add_argument("--per_device_train_batch_size", type=int, default=8, # A100 GPU 4096 max_seq_length
                    help="Batch size per device.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of gradient accumulation steps.")
parser.add_argument("--perc", type=float, default=1.0,
                    help="Percentage of the training dataset to use.")

# LoRA Parameters:
parser.add_argument("--lora_r", type=float, default=32,
                    help=r"Dimensionality of low-rank matrices used for \delta W.")
parser.add_argument("--lora_alpha", type=float, default=64,
                    help=r"Weight used on update \lora_alpha * \deltaW.")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="Dropout parameter for LoRA adapter layers.")

parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

MODEL_NAME = args.model_name

NUM_TARGET_SAMPLES, NUM_PARAPHRASES = os.path.basename(args.dataset_path).split("_")[-1].split("-")
NUM_TARGET_SAMPLES = int(NUM_TARGET_SAMPLES)
NUM_PARAPHRASES = int(NUM_PARAPHRASES)
print("NUM_TARGET_SAMPLES={}".format(NUM_TARGET_SAMPLES))
print("NUM_PARAPHRASES={}".format(NUM_PARAPHRASES))

def get_max_seq_length():
    # 8 * 4 posts * 128 tokens
    # 4 posts = original + 3 paraphrases
    # 3 paraphrases empirically found to be the best
    # extra 128 for buffer
    if args.max_seq_length is not None:
        return args.max_seq_length
    max_seq_len = NUM_TARGET_SAMPLES * (NUM_PARAPHRASES + 1) * 128
    return max_seq_len
    
def get_device_string():
    device_string = PartialState().process_index 
    return device_string

def load_model_and_tokenizer():
    device_string = get_device_string()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map={'':device_string},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    # https://github.com/huggingface/peft/issues/137
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none", # keeps the original model perform equally when the adapter is not used
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        add_eos_token=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dataset() -> tuple[list[str, list[str]]]:
    train_dataset = load_from_disk(args.dataset_path)
    train_dataset = train_dataset.shuffle()
    if args.debug:
        train_dataset = train_dataset.select(range(100))
    return train_dataset

def get_experiment_dir() -> str:
    dataset_name = os.path.basename(args.dataset_path)
    dataset_name = os.path.basename(MODEL_NAME) + "--" + dataset_name
    if args.debug:
        dataset_name += "_debug"
    experiment_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def get_run_name() -> str:
    run_name = f"r={args.lora_r}_alpha={args.lora_alpha}_dropout={args.lora_dropout}_perc={args.perc}"
    return run_name

def save_hparams(experiment_dir: str, run_name: str) -> None:
    hparams = OrderedDict(
        dataset_path=args.dataset_path,
        lr=args.lr,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        perc=args.perc,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    os.makedirs(os.path.join(experiment_dir, run_name), exist_ok=True)
    with open(os.path.join(experiment_dir, run_name, "hparams.json"), "w") as fout:
        json.dump(hparams, fout, indent=4)
        
def main():
    if args.debug:
        print(colored("Running in DEBUG mode", "green"))

    for key, value in vars(args).items():
        print(colored(f"\t{key} = {value}", "yellow"))
    
    train_samples = load_dataset()
    if "mistral" in os.path.basename(MODEL_NAME).lower():
        response_template = "[/INST] Original:"
    elif "llama" in os.path.basename(MODEL_NAME).lower():
        response_template = "==== Original:"
    else:
        assert False
    print("Response Template: {}".format(response_template))
    
    print(colored(f"len(train_samples)={len(train_samples)}", "yellow"))

    experiment_dir = get_experiment_dir()
    run_name = get_run_name()
    save_hparams(experiment_dir, run_name)
    output_dir = os.path.join(experiment_dir, run_name)
    
    model, tokenizer = load_model_and_tokenizer()

    max_steps = 20 if args.debug else args.max_steps
    save_steps = 10 if args.debug else args.save_steps
    logging_steps = 1 if args.debug else args.logging_steps
    print("AM I DEBUGGING? ", args.debug)

    warmup_steps = int(ceil(len(train_samples) / PartialState().num_processes) * 0.01)
    config = SFTConfig(
        max_seq_length=get_max_seq_length(),
        report_to="tensorboard",
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant" : False},
        optim="adamw_torch",
        bf16=True,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        num_train_epochs=args.num_train_epochs,
        # max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        # https://discuss.huggingface.co/t/training-llama-with-lora-on-multiple-gpus-may-exist-bug/47005/3
        ddp_find_unused_parameters=False,
    )
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer,
    )
    
    trainer = SFTTrainer(
        args=config,
        model=model,
        train_dataset=train_samples,
        data_collator=collator,
    )
    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

if __name__ == "__main__":
    assert args.perc > 0.0 and args.perc <= 1.0
    sys.exit(main())