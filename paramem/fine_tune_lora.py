import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, TrainingArguments, ProgressCallback, EarlyStoppingCallback
from trl import SFTTrainer
import pandas as pd
import typer
from pathlib import Path
from datasets import Dataset
import logging
from accelerate import Accelerator
from data import load_csv_data, load_pile_data
from accelerate import load_checkpoint_and_dispatch
from peft import get_peft_model, LoraConfig, TaskType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from paramem.evaluation.utils import homemade_data_files
import random

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train_model(model, lr=None, callbacks=None, tr_dataset=None, va_dataset=None, epochs=1000, batch_size=5, output_dir="./models", eval_steps=50, max_saved_ckpts=None):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )    

    model = get_peft_model(model, peft_config)


    trainer = SFTTrainer(
        model=model,
        train_dataset=tr_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            report_to='wandb',
            do_eval=True,
            eval_steps=eval_steps, eval_strategy="steps",
            load_best_model_at_end=True,
            dataloader_drop_last=True,
            save_total_limit=max_saved_ckpts,
        ),
        eval_dataset=va_dataset,
        dataset_text_field="text",
        callbacks=callbacks
    )
    trainer.train()
    
    unwrapped_model = trainer.accelerator.unwrap_model(model)
    # del model
    # torch.cuda.empty_cache()
    # unwrapped_model.save_pretrained(output_dir, max_shard_size='10GB')
    # Save only the LoRA weights
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state = trainer.accelerator.get_state_dict(unwrapped_model)
        # lora_weights = {name: param for name, param in state if "lora" in name}
        torch.save(state, Path(output_dir) / "lora_weights.pth") 
    return unwrapped_model

def main(
    model_name: str="mistralai/Mistral-7B-v0.3",
    train_file: str="./data3/wikidata_Mis7.csv",
    output_dir: str="./models",
    batch_size: int=1, 
    train_frac: float=0.8,
    save_inputs: bool=False,
    lr: float=1e-5,
    epochs: int=10,
    seed: int=42,
    eval_steps: int=100,
    max_saved_ckpts:int=10,
    overwrite: bool=False,
    ):
    torch.manual_seed(seed)
    if Path(output_dir).exists() and not overwrite:
        checkpoints=sorted(Path(output_dir).glob("checkpoint-*"))
        if len(checkpoints)>0:
            model_name = checkpoints[-1]
    else:
        # delete all checkpoints
        for ckpt in Path(output_dir).glob("checkpoint-*"):
            if ckpt.is_dir():
                for file in ckpt.glob("*"):
                    file.unlink()
                ckpt.rmdir()
            else:
                ckpt.unlink()
        model_name = model_name
            
    log_file=f"{output_dir}/training.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file)
    logging.getLogger().setLevel(logging.INFO)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = homemade_data_files(train_file, seed=seed, train_frac=train_frac, output_dir=output_dir)
    tr_dataset = dataset["train"]
    va_dataset = dataset["validation"]
    callbacks = [
        ProgressCallback(),
        EarlyStoppingCallback(early_stopping_patience=max_saved_ckpts, ),
    ]
    callbacks=None

    model = train_model(model, lr=lr, callbacks=callbacks, tr_dataset=tr_dataset, va_dataset=va_dataset, epochs=epochs, batch_size=batch_size, output_dir=output_dir, eval_steps=eval_steps, max_saved_ckpts=max_saved_ckpts)
    
    # model.to("cpu")
    # model.save_pretrained(output_dir)

def load_lora_weights(model, lora_weights_path):
    lora_weights = torch.load(lora_weights_path)
    model.load_state_dict(lora_weights, strict=False)
    return model

if __name__ == "__main__":
    typer.run(main)