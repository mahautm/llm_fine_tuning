import os

import typer
from typing import Optional
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import logging as hf_logging
from paramem.evaluation.utils import homemade_data_files
from pathlib import Path

import subprocess
# Note: Removed DeepSpeedCPUAdam import due to compilation issues
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
import socket

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)

app = typer.Typer()

class DeepSpeedTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps):
        # Let DeepSpeed handle optimizer creation
        if self.deepspeed:
            return
        super().create_optimizer_and_scheduler(num_training_steps)

def maybe_init_distributed():
    """Initialize distributed training if environment variables are set"""
    if dist.is_available() and not dist.is_initialized():
        # Only initialize if we have the required environment variables
        rank = os.environ.get("RANK", None)
        world_size = os.environ.get("WORLD_SIZE", None)
        
        if rank is not None and world_size is not None:
            try:
                # Set timeout to handle hanging processes
                timeout = 3600  # 1 hour timeout
                dist.init_process_group(
                    backend="nccl", 
                    init_method="env://",
                    timeout=torch.distributed.constants.default_pg_timeout if hasattr(torch.distributed, 'constants') else None
                )
                print(f"Initialized distributed training: rank={rank}, world_size={world_size}")
            except Exception as e:
                print(f"Failed to initialize distributed training: {e}")
                print("Falling back to single-GPU mode")
        else:
            print("Distributed environment variables not found, running in single-GPU mode")

DATASETS = {
    "wikitext": ("wikitext", "wikitext-2-raw-v1"),
    "c4": ("c4", "en"),
    "openwebtext": ("openwebtext", "openwebtext"),
    "bookcorpus": ("bookcorpus", "bookcorpus"),
    }

USING_LORA = False
BASE_MODEL_NAME = ""


def _infer_lora_target_modules(model) -> list[str]:
    module_names = [name for name, _ in model.named_modules()]

    # LLaMA/Mistral-style attention projections.
    if any(name.endswith("q_proj") for name in module_names) and any(
        name.endswith("v_proj") for name in module_names
    ):
        return ["q_proj", "v_proj"]

    # GPT-NeoX/Pythia uses fused QKV projection.
    if any(name.endswith("query_key_value") for name in module_names):
        return ["query_key_value"]

    # GPT-2 style fused projection fallback.
    if any(name.endswith("c_attn") for name in module_names):
        return ["c_attn"]

    # Conservative generic fallback for unknown architectures.
    return ["q_proj", "v_proj"]

class CheckpointNotifyCallback(TrainerCallback):
    """
    Callback to notify when a checkpoint is saved.
    This can be used to trigger a script or notification.
    """
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        # Launch a bash job, passing the checkpoint directory as an environment variable
        # this is how we launch all testing on the given checkpoint.
        # Use EVAL_SCRIPT env var if set, otherwise use default launch_checkpoint_tests.sh
        eval_script = os.environ.get("EVAL_SCRIPT", "/home/mmahaut/projects/paramem/scripts/launch_checkpoint_tests.sh")
        subprocess.Popen(
            ["bash", eval_script],
            env={
                **os.environ,
                "CHECKPOINT_PATH": checkpoint_dir,
                "USE_LORA": str(int(USING_LORA)),
                "MODEL_NAME": BASE_MODEL_NAME,
            },
        )
        # Or log/notify as needed
        typer.echo(f"Checkpoint saved at {checkpoint_dir}. Test script launched with {eval_script}.")

def get_model(
    model_name: str,
    use_lora: bool,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    checkpoint_path: Optional[str] = None,
    fsdp: bool = False,
):
    device_map = None #if fsdp else ("auto" if torch.cuda.is_available() else None)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path if checkpoint_path else model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )
    if use_lora:
        target_modules = _infer_lora_target_modules(model)
        print(f"Using LoRA target modules: {target_modules}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,  # Ensure training mode
        )
        model = get_peft_model(model, lora_config)
        # Print trainable parameters for debugging
        model.print_trainable_parameters()
    return model

def get_tokenizer(model_name: str):
    tokenizer=AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # if needed add pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_dataset(dataset_name: str, split: str = "train", n_samples: Optional[int] = None, output_dir: Optional[str] = None, seed: int = 42) -> Dataset:
    if dataset_name.endswith(".csv") or dataset_name.endswith(".txt"):
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        dataset = homemade_data_files(dataset_name, seed=seed, train_frac=0.8, output_dir=output_dir)[split]

        # DEDUPLICATION: Remove duplicates based on "text" column if it exists
        if "text" in dataset.column_names:
            seen_texts = set()
            def is_unique(example):
                text = example["text"]
                if text in seen_texts:
                    return False
                seen_texts.add(text)
                return True
            dataset = dataset.filter(is_unique)
        else:
            # Remove duplicates based on all columns
            seen_rows = set()
            def is_unique_row(example):
                row_tuple = tuple(example.values())
                if row_tuple in seen_rows:
                    return False
                seen_rows.add(row_tuple)
                return True
            dataset = dataset.filter(is_unique_row)
    else:
        # Use DATASETS mapping if available
        if dataset_name in DATASETS:
            ds_args = DATASETS[dataset_name]
            dataset = load_dataset(*ds_args, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)

    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
    return dataset

def tokenize_function(examples, tokenizer, block_size):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=block_size,
        return_special_tokens_mask=True,
    )

def prepare_dataset(dataset, tokenizer, block_size):
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer, block_size),
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized

def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

@app.command()
def main(
    model_name: str = typer.Option(..., help="HuggingFace model name"),
    dataset_name: str = typer.Option(..., help="HuggingFace dataset name"),
    output_dir: str = typer.Option("./finetuned_model", help="Output directory"),
    use_lora: bool = typer.Option(False, help="Use LoRA finetuning"),
    lora_r: int = typer.Option(8, help="LoRA rank"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.1, help="LoRA dropout"),
    block_size: Optional[int] = typer.Option(None, help="Block size for tokenization"),
    per_device_train_batch_size: int = typer.Option(4, help="Batch size per device"),
    num_train_epochs: int = typer.Option(3, help="Number of training epochs"),
    save_steps: int = typer.Option(500, help="Save checkpoint every N steps"),
    fsdp: bool = typer.Option(False, help="Use FSDP for distributed training"),
    deep_speed: bool = typer.Option(False, help="Use DeepSpeed for training"),
    checkpoint_path: Optional[str] = typer.Option(None, help="Path to resume checkpoint"),
    resume_from_checkpoint: Optional[str] = typer.Option(None, help="Resume from checkpoint"),
    logging_steps: int = typer.Option(50, help="Logging steps"),
    gradient_accumulation_steps: int = typer.Option(1, help="Gradient accumulation steps"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
    n_samples: int = typer.Option(None, help="Number of samples to use from the dataset. If None, use the full dataset."),
    eval_script: Optional[str] = typer.Option(None, help="Path to checkpoint evaluation script. If not set, uses EVAL_SCRIPT env var or default."),
):
    maybe_init_distributed()
    
    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set CUDA device safely
    if torch.cuda.is_available():
        if local_rank >= 0 and local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
        else:
            torch.cuda.set_device(0)  # Default to device 0
            local_rank = 0

    hf_logging.set_verbosity_debug()
    node_id = socket.gethostname()
    print(f"[{node_id}]: pid={os.getpid()}, local_rank={local_rank}, device={torch.cuda.current_device()}, visible={torch.cuda.device_count()}, SLURM_NODEID={os.environ.get('SLURM_NODEID', 'N/A')}, torchrun_rank={os.environ.get('RANK', 'N/A')}")
    tokenizer = get_tokenizer(model_name)
    raw_dataset = get_dataset(dataset_name, n_samples=n_samples, output_dir=output_dir)

    # If block_size is not set, use max sentence size from dataset
    if block_size is None:
        # Assume "text" column exists
        max_sentence_size = max(len(tokenizer.encode(x["text"])) for x in raw_dataset)
        block_size = max_sentence_size
        typer.echo(f"Auto-detected block_size from dataset: {block_size}")
    # maintain one batch on GPU before building model
    # dummy_batch = torch.zeros((per_device_train_batch_size*10, block_size), dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Set EVAL_SCRIPT environment variable if provided
    if eval_script:
        os.environ["EVAL_SCRIPT"] = eval_script
        typer.echo(f"Using custom evaluation script: {eval_script}")
    
    typer.echo("Finetuning parameters:")
    typer.echo(f"  model_name: {model_name}")
    typer.echo(f"  dataset_name: {dataset_name}")
    typer.echo(f"  output_dir: {output_dir}")
    typer.echo(f"  use_lora: {use_lora}")
    typer.echo(f"  lora_r: {lora_r}")
    typer.echo(f"  lora_alpha: {lora_alpha}")
    typer.echo(f"  lora_dropout: {lora_dropout}")
    typer.echo(f"  block_size: {block_size}")
    typer.echo(f"  per_device_train_batch_size: {per_device_train_batch_size}")
    typer.echo(f"  num_train_epochs: {num_train_epochs}")
    typer.echo(f"  save_steps: {save_steps}")
    typer.echo(f"  fsdp: {fsdp}")
    typer.echo(f"  deep_speed: {deep_speed}")
    typer.echo(f"  checkpoint_path: {checkpoint_path}")
    typer.echo(f"  resume_from_checkpoint: {resume_from_checkpoint}")
    typer.echo(f"  logging_steps: {logging_steps}")
    typer.echo(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
    typer.echo(f"  learning_rate: {learning_rate}")
    typer.echo(f"  n_samples: {n_samples}")

    train_dataset = prepare_dataset(raw_dataset, tokenizer, block_size)
    global USING_LORA
    global BASE_MODEL_NAME
    USING_LORA = use_lora
    BASE_MODEL_NAME = model_name

    model = get_model(
        model_name,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
        checkpoint_path,
        fsdp or deep_speed,
    )
    # delete dummy batch to free memory
    # del dummy_batch

    data_collator = get_data_collator(tokenizer)

    # FSDP parameters you can tune in TrainingArguments (transformers >=4.30):
    # - fsdp: FSDP mode string, e.g. "full_shard auto_wrap"
    # - fsdp_config: dict with FSDP options, e.g. {
    #       "activation_checkpointing": True,
    #       "min_num_params": 1e8,
    #       "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
    #       "cpu_offload": True,
    #       "mixed_precision": True,
    #       "forward_prefetch": True,
    #       "state_dict_type": "full",
    #       "sync_module_states": True,
    #       "use_orig_params": True,
    #   }
    # See: https://huggingface.co/docs/transformers/main/en/fsdp

    class ClearCUDACacheCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Optimize batch size and gradient accumulation based on training type
    effective_batch_size = per_device_train_batch_size if use_lora else max(1, per_device_train_batch_size // 2)
    effective_grad_accum = gradient_accumulation_steps if use_lora else gradient_accumulation_steps * 2
    
    print(f"🔧 Memory Optimization Settings:")
    print(f"   Training Mode: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    print(f"   Batch Size: {per_device_train_batch_size} → {effective_batch_size}")
    print(f"   Gradient Accumulation: {gradient_accumulation_steps} → {effective_grad_accum}")
    print(f"   Gradient Checkpointing: {not use_lora}")
    print(f"   DeepSpeed Config: ds_config_{'lora' if use_lora else 'full'}.json")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=effective_batch_size,
        save_steps=save_steps,
        save_total_limit=None,
        logging_steps=logging_steps,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=learning_rate,
        fp16=False,  # bf16 is set above, so disable fp16
        report_to=["wandb"] if not deep_speed else None,  # Enable wandb logging
        ddp_find_unused_parameters=False,
        gradient_checkpointing=not use_lora,  # Enable for full fine-tuning, disable for LoRA
        dataloader_drop_last=True,  # Important for distributed training
        # Use memory-optimized DeepSpeed config based on training type
        deepspeed=f"/home/mmahaut/projects/paramem/scripts/ds_config_{'lora' if use_lora else 'full'}_fixed.json" if deep_speed else None,
        bf16=True,
        skip_memory_metrics=True,  # Reduce memory overhead
    )
    callbacks = [CheckpointNotifyCallback()] 

    # Ensure model is in training mode and parameters require grad
    model.train()
    
    # Debug: Check model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    typer.echo(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    if use_lora:
        # Ensure LoRA parameters require gradients
        lora_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.requires_grad_(True)
                if "lora" in name.lower():
                    lora_params += param.numel()
        typer.echo(f"LoRA parameters: {lora_params:,}")
        
        # Debug: Print some parameter names
        grad_params = [name for name, param in model.named_parameters() if param.requires_grad]
        typer.echo(f"First few parameters requiring grad: {grad_params[:5]}")
    
    trainer = DeepSpeedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        typer.echo(f"Finetuning complete. Model saved to {output_dir}")
    except Exception as e:
        typer.echo(f"Training failed with error: {e}")
        raise
    finally:
        # Proper cleanup to prevent hanging processes
        try:
            # Save any pending checkpoints
            if 'trainer' in locals():
                try:
                    trainer.save_state()
                except Exception:
                    pass
            
            # Clear CUDA cache before destroying process group
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Destroy process group with timeout
            if dist.is_initialized():
                try:
                    # Give processes time to finish
                    import time
                    time.sleep(1)
                    dist.destroy_process_group()
                    typer.echo("Successfully cleaned up distributed training")
                except Exception as cleanup_error:
                    typer.echo(f"Warning: Error during process group cleanup: {cleanup_error}")
                    # Force cleanup
                    os._exit(0)
                    
        except Exception as final_cleanup_error:
            typer.echo(f"Warning: Error during final cleanup: {final_cleanup_error}")


if __name__ == "__main__":
    app()

    # Example call:
    # python /home/mmahaut/projects/paramem/paramem/fine_tune_with_ckpts.py \
    #   --model-name "mistralai/Mistral-7B-v0.3" \
    #   --dataset-name "wikitext" \
    #   --output-dir "./finetuned_Mis7" \
    #   --use-lora \
    #   --lora-r 4 \
    #   --lora-alpha 16 \
    #   --lora-dropout 0.05 \
    #   --block-size 128 \
    #   --per-device-train-batch-size 2 \
    #   --num-train-epochs 1 \
    #   --save-steps 100 \
    #   --logging-steps 10 \
    #   --gradient-accumulation-steps 2 \
    #   --learning-rate 1e-4 \
    #   --n-samples 200

    # Example call to resume training from a previous checkpoint:
    # python /home/mmahaut/projects/paramem/paramem/fine_tune_with_ckpts.py \
    #   --model-name "mistralai/Mistral-7B-v0.3" \
    #   --dataset-name "wikitext" \
    #   --output-dir "./finetuned_Mis7" \
    #   --use-lora \
    #   --lora-r 4 \
    #   --lora-alpha 16 \
    #   --lora-dropout 0.05 \
    #   --block-size 128 \
    #   --per-device-train-batch-size 2 \
    #   --num-train-epochs 1 \
    #   --save-steps 100 \
    #   --logging-steps 10 \
    #   --gradient-accumulation-steps 2 \
    #   --learning-rate 1e-4 \
    #   --n-samples 200 \
    #   --resume-from-checkpoint "./finetuned_Mis7/checkpoint-100"

    # Example call with FSDP distributed training:
    # torchrun --nproc_per_node=2 /home/mmahaut/projects/paramem/paramem/fine_tune_with_ckpts.py \
    #   --model-name "mistralai/Mistral-7B-v0.3" \
    #   --dataset-name "wikitext" \
    #   --output-dir "./finetuned_Mis7_fsdp" \
    #   --use-lora \
    #   --lora-r 4 \
    #   --lora-alpha 16 \
    #   --lora-dropout 0.05 \
    #   --block-size 128 \
    #   --per-device-train-batch-size 2 \
    #   --num-train-epochs 1 \
    #   --save-steps 100 \
    #   --logging-steps 10 \
    #   --gradient-accumulation-steps 2 \
    #   --learning-rate 1e-4 \
    #   --n-samples 200 \
    #   --fsdp