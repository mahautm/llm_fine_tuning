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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# freeze all layers except selected ones
def freeze_layers(model, layers_to_unfreeze):
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the selected layers
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

# Example usage:
# Assuming you want to unfreeze the last 5 transformer blocks
def get_layers_to_unfreeze(model):
    model_name = model.__class__.__name__.lower()
    if "mistral" in model_name:
        return model.model.layers[5:]
    else:
        raise ValueError(f"Model {model_name} not supported for layer unfreezing")


def train_model(model, lr=None, callbacks=None, tr_dataset=None, va_dataset=None, epochs=1000, batch_size=5, output_dir="./models", accelerator=None, eval_steps=50, max_saved_ckpts=None):
    
    # freeze
    layers_to_unfreeze = get_layers_to_unfreeze(model)
    freeze_layers(model, layers_to_unfreeze)

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
            # gradient_checkpointing=True,
            # gradient_accumulation_steps=4,
        ),
        eval_dataset=va_dataset,
        # optimizers=(optimizer, None),
        dataset_text_field="text",
        callbacks=callbacks
    )
    trainer.train()
    # save the model
    trainer.save_model(output_dir)
    return model

def main(
    model_name: str="mistralai/Mistral-7B-Instruct-v0.3",
    train_file: str="./data/wikidata_incl_m7i.csv",
    output_dir: str="./models",
    batch_size: int=1, 
    train_frac: float=0.8,
    save_inputs: bool=False,
    lr: float=1e-5,
    epochs: int=10,
    seed: int=42,
    eval_steps: int=100,
    max_saved_ckpts:int=3,
    # use_accelerator: bool=True
    ):
    # max_saved_ckpts is used in TrainerArgs as save_total_limit, and in the early_stopping as patience
    # seed
    torch.manual_seed(seed)
    if Path(output_dir).exists():
        # latest number in Path
        checkpoints=sorted(Path(output_dir).glob("checkpoint-*"))
        if len(checkpoints)>0:
            model_name = checkpoints[-1]
            
    # logging
    log_file=f"{output_dir}/training.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file)
    logging.getLogger().setLevel(logging.INFO)

    # if use_accelerator:
    #     accelerator = Accelerator()
    #     batch_size = batch_size * accelerator.num_processes
    # else:
    #     accelerator = None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map= "balanced",# if not use_accelerator else None,
        # torch_dtype=torch.float16
        )


    ## DATA preparation
    if ".csv" in train_file:
        inputs = load_csv_data(train_file, "query", False, threshold_knowledge=False)
        df = pd.DataFrame(inputs)
        df = df.dropna()
        df["expected_answers"]=df["expected_answers"].apply(pd.eval)
        df = df.explode("expected_answers")
        df["text"] = df.apply(lambda x: x["query"] + " " + str(x["expected_answers"]), axis=1)

    elif ".txt" in train_file:
        inputs = load_pile_data(train_file, "text")
        df = pd.DataFrame(columns=["text"], data=inputs)

    # train test split
    tr_df = df.sample(frac=train_frac, random_state=seed)[["text"]]
    te_df = df.drop(tr_df.index)[["text"]]
    # half for test half for validation
    te_len=te_df.shape[0]//2
    va_df = te_df[:te_len]
    te_df = te_df[te_len:]

    if save_inputs:
        df.to_csv(f"{output_dir}/data.csv", index=False)
        tr_df.to_csv(f"{output_dir}/train.csv", index=False)
        te_df.to_csv(f"{output_dir}/test.csv", index=False)
        va_df.to_csv(f"{output_dir}/val.csv", index=False)
    tr_dataset = Dataset.from_pandas(tr_df)
    va_dataset = Dataset.from_pandas(va_df)
    ## End of data preparation

    ## CALLBACKS
    # training callbacks
    callbacks = [
        ProgressCallback(),
        EarlyStoppingCallback(early_stopping_patience=max_saved_ckpts, ),
    ]
    callbacks=None
    ## END OF CALLBACKS 

    ## TRAIN + SAVE model
    model = train_model(model, lr=lr, callbacks=callbacks, tr_dataset=tr_dataset, va_dataset=va_dataset, epochs=epochs, batch_size=batch_size, output_dir=output_dir, eval_steps=eval_steps, max_saved_ckpts=max_saved_ckpts)#, accelerator=accelerator)
    ## END OF TRAINING
    

if __name__ == "__main__":
    typer.run(main)