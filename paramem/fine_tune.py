import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, ProgressCallback, EarlyStoppingCallback
from trl import SFTTrainer
import pandas as pd
import typer
from pathlib import Path
from datasets import Dataset
import logging
from accelerate import Accelerator
from data import load_csv_data, load_pile_data


def train_model(model, lr=None, callbacks=None, tr_dataset=None, va_dataset=None, epochs=1000, batch_size=5, output_dir="./models", accelerator=None, eval_steps=100, max_saved_ckpts=None):
    # optimizer = AdamW(model.parameters(), lr=lr)
    # if accelerator is not None:
    #     logging.info("Using accelerator")
    #     model, optimizer, tr_dataloader, va_dataloader = accelerator.prepare(model, optimizer, tr_dataloader, va_dataloader)

    trainer = SFTTrainer(
        model=model,
        train_dataset=tr_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, 
            eval_steps=eval_steps, eval_strategy="steps",
            load_best_model_at_end=True,
            dataloader_drop_last=True,
            save_total_limit=max_saved_ckpts
        ),
        eval_dataset=va_dataset,
        # optimizers=(optimizer, None),
        dataset_text_field="text",
        callbacks=callbacks
    )
    trainer.train()
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
    max_saved_ckpts=3,
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
        device_map=None#"auto" if not use_accelerator else None,
        # dtype=torch.bfloat16 if use_accelerator else None
    )

    ## DATA preparation
    if ".csv" in train_file:
        df = load_csv_data(train_file, "query", False)
        df["expected_answers"] = df["expected_answers"].apply(lambda x: x[0])
        df["text"] = df.apply(lambda x: x["query"] + " " + x["expected_answers"], axis=1)

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
    ## END OF CALLBACKS 

    ## TRAIN
    model = train_model(model, lr=lr, callbacks=callbacks, tr_dataset=tr_dataset, va_dataset=va_dataset, epochs=epochs, batch_size=batch_size, output_dir=output_dir, eval_steps=eval_steps, max_saved_ckpts=max_saved_ckpts)#, accelerator=accelerator)
    ## END OF TRAINING
    
    ## SAVE
    model.save_pretrained(output_dir)
    ## END OF SAVE

if __name__ == "__main__":
    typer.run(main)