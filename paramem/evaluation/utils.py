from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset, Dataset
from typing import List, Dict, Any
import json
from pathlib import Path
import torch
import pandas as pd
import random
from paramem.data import load_csv_data, load_pile_data

# Import deepspeed lazily / safely: some environments don't have CUDA_HOME set or
# require compiled ops. We'll attempt to import here but fall back to None and
# raise a helpful error only when DeepSpeed functionality is required.
try:
    import deepspeed  # type: ignore
    from transformers import AutoConfig
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    _deepspeed_import_error = None
except Exception as _e:
    deepspeed = None  # type: ignore
    load_state_dict_from_zero_checkpoint = None  # type: ignore
    _deepspeed_import_error = _e

# List of benchmark datasets (can be expanded)
BENCHMARK_DATASETS = [
    {"name": "mmlu", "subset": "all"},  # Massive Multitask Language Understanding
    {"name": "arc", "subset": "challenge"},  # AI2 Reasoning Challenge
    {"name": "hellaswag", "subset": None},  # Commonsense reasoning
    {"name": "truthful_qa", "subset": "mc"},  # TruthfulQA multiple choice
    {"name": "gsm8k", "subset": "main"},  # Grade School Math
    {"name": "winogrande", "subset": "winogrande_xs"},  # Commonsense reasoning
    {"name": "openbookqa", "subset": "main"},  # OpenBookQA
    {"name": "lambada", "subset": "plain_text"},  # LAMBADA word prediction
]

def load_model(checkpoint_path: str, use_lora: bool = False) -> Any:
    """
    Loads a model and tokenizer from a checkpoint.
    Args:
        checkpoint_path (str): Path to the checkpoint directory.
        use_lora (bool): Whether the model was fine-tuned with LoRA.
    Returns:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist.")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    # if needed add Padding and special tokens
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
        

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {checkpoint_path} on device {device_map}...")

    # Check for DeepSpeed checkpoint (e.g., presence of 'zero_to_fp32.py' or 'ds_inference_config.json')
    deepspeed_files = ["zero_to_fp32.py", "ds_inference_config.json"]
    has_deepspeed = any((Path(checkpoint_path) / f).exists() for f in deepspeed_files)

    if has_deepspeed:
        # check if the merged model already exists
        merged_model_path = Path(checkpoint_path) / "merged_single_model"
        if merged_model_path.exists():
            print(f"Loading merged model from {merged_model_path}")
            model = AutoModelForCausalLM.from_pretrained(merged_model_path, device_map=device_map)
        else:
            try:
                # Find the base model config
                config = AutoConfig.from_pretrained(checkpoint_path)
                model = AutoModelForCausalLM.from_config(config)
                # Use DeepSpeed to load the full weights into one model
                # This assumes zero_to_fp32.py is present and can be used
                step_number = checkpoint_path.split("-")[-1]
                zero_ckpt = Path(checkpoint_path) / f"global_step{step_number}" /"zero_pp_rank_0_mp_rank_00_model_states.pt"
                if not zero_ckpt.exists():
                    raise FileNotFoundError(f"DeepSpeed checkpoint not found at {zero_ckpt}")
                # Use DeepSpeed's utility to merge shards (requires deepspeed installed)
                if load_state_dict_from_zero_checkpoint is None:
                    # Provide a clearer message about why import failed
                    raise ImportError(
                        "DeepSpeed utilities are not available. "
                        "This can happen if DeepSpeed failed to import (e.g. CUDA_HOME not set, or ops not compiled). "
                        f"Original error: {_deepspeed_import_error}"
                    )
                load_state_dict_from_zero_checkpoint(model, checkpoint_path)
                print("Loaded DeepSpeed model into a single model.")
                # Save the merged model as a single checkpoint for future use
                single_model_path = Path(checkpoint_path) / "merged_single_model"
                single_model_path.mkdir(exist_ok=True)
                model.save_pretrained(single_model_path)
                tokenizer.save_pretrained(single_model_path)
                print(f"Saved merged model and tokenizer to {single_model_path}")
                model = AutoModelForCausalLM.from_pretrained(single_model_path, device_map=device_map)
            except ImportError:
                raise ImportError("deepspeed must be installed to load DeepSpeed checkpoints.")
    else:
        if use_lora:
            # Load PEFT config and base model, then apply LoRA weights
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map=device_map)
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map=device_map)
    model.eval()
    return model, tokenizer

def homemade_data_files(train_file: str, seed:int, train_frac:float=0.8, output_dir:str=None) -> Dict[str, List[str]]:
    if ".csv" in train_file:
        df = pd.read_csv(train_file, dtype=str)
        df = df.dropna()
        rng = random.Random(seed)
        if "query" in df.columns and "expected_answers" in df.columns:
            df["expected_answers"] = df["expected_answers"].apply(
                lambda x: pd.eval(x)[rng.randint(0, len(pd.eval(x)) - 1)]
            )
            df["text"] = df.apply(lambda x: x["query"] + " " + str(x["expected_answers"]), axis=1)
            # still haven't found why the quotes sometimes appear in the first 4 characters, removing
            df["text"] = df["text"].apply(lambda x: x.replace("\" ", "").replace(" \"", "") if "\" " in x[:4] else x)
            print(f"There are {df.shape[0]} samples in the dataset")
        elif "text" in df.columns:
            df = df[["text"]]
            print(f"There are {df.shape[0]} samples in the dataset")
        else:
            raise ValueError("CSV file must contain 'query' and 'expected_answers' or 'text' columns.")

    elif ".txt" in train_file:
        inputs = load_pile_data(train_file, "text")
        df = pd.DataFrame(columns=["text"], data=inputs)

    # some rows have text between quotes, removing them
    df["text"] = df["text"].apply(
        lambda x: x[1:-1] if x.startswith("\"") and x.endswith("\"") else x
    )
    df["text"] = df["text"].apply(lambda x: x[1:] if x.startswith(" ") else x)
    tr_df = df.sample(frac=train_frac, random_state=seed)[["text"]]
    te_df = df.drop(tr_df.index)[["text"]]
    te_len=te_df.shape[0]//2
    va_df = te_df[:te_len]
    te_df = te_df[te_len:]

    if output_dir is not None:
        df.to_csv(f"{output_dir}/data.csv", index=False)
        tr_df.to_csv(f"{output_dir}/train.csv", index=False)
        te_df.to_csv(f"{output_dir}/test.csv", index=False)
        va_df.to_csv(f"{output_dir}/val.csv", index=False)
    tr_dataset = Dataset.from_pandas(tr_df)
    va_dataset = Dataset.from_pandas(va_df)
    te_df = Dataset.from_pandas(te_df)
    return {
        "train": tr_dataset,
        "validation": va_dataset,
        "test": te_df
    }