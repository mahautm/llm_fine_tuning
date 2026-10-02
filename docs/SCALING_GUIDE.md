# Setup Guide: Training Bigger Models

This guide helps you scale up from Llama-3.2-1B to larger models (7B, 13B, 70B+) using the existing fine-tuning infrastructure.

## Existing Infrastructure

### 1. Fine-Tuning Scripts

**Main Training Script**: `paramem/fine_tune_with_ckpts.py`
- Supports both **Full Fine-Tuning** and **LoRA**
- Features:
  - FSDP (Fully Sharded Data Parallel) for multi-GPU training
  - DeepSpeed integration
  - Automatic checkpoint saving and testing
  - Resume from checkpoints
  - Gradient accumulation
  - Mixed precision (bfloat16)

**SLURM Launcher**: `scripts/fine_tune_multi_node.sh`
- Handles multi-node distributed training
- Automatically configures memory based on training mode
- Supports multiple datasets and learning rates

**Worker Script**: `scripts/run_ft_worker.sh`
- Executes training on individual nodes
- Uses `torchrun` for distributed coordination

### 2. Current Model Setup (Llama-3.2-1B)

**Configuration Used:**
- Batch size: 5 per device
- Learning rate: 1e-4
- LoRA config: r=8, alpha=32, dropout=0.1
- Memory: 400GB (Full FT), 100GB (LoRA)
- Target modules: ["q_proj", "v_proj"]

## Scaling to Bigger Models

### Memory Requirements (Approximate)

| Model Size | Full FT | LoRA | Inference |
|------------|---------|------|-----------|
| 1B params  | 400GB   | 100GB| 10GB      |
| 7B params  | 2.8TB   | 700GB| 70GB      |
| 13B params | 5.2TB   | 1.3TB| 130GB     |
| 70B params | 28TB    | 7TB  | 700GB     |

**Note**: These are conservative estimates including optimizer states and gradients.

### Recommended Configurations

#### 7B Models (e.g., Llama-2-7B, Mistral-7B)

**Full Fine-Tuning:**
```bash
# Requires 4-8 A100 80GB GPUs
sbatch scripts/fine_tune_multi_node.sh

# Parameters to adjust in script:
MEM_FULL="700G"
BATCH_SIZE=1
GRAD_ACCUM=8
NUM_GPUS=8
```

**LoRA (Recommended):**
```bash
# Can run on 1-2 A100 80GB GPUs
MEM_LORA="150G"
BATCH_SIZE=4
LORA_R=16
LORA_ALPHA=32
NUM_GPUS=2
```

#### 13B Models (e.g., Llama-2-13B)

**LoRA Only (Recommended):**
```bash
# Requires 4 A100 80GB GPUs
MEM_LORA="300G"
BATCH_SIZE=2
LORA_R=16
LORA_ALPHA=32
NUM_GPUS=4
LORA_TARGET_MODULES=["q_proj", "v_proj", "k_proj", "o_proj"]
```

#### 70B Models (e.g., Llama-2-70B)

**LoRA with Quantization (Recommended):**
```bash
# Requires 8 A100 80GB GPUs
MEM_LORA="800G"
BATCH_SIZE=1
GRAD_ACCUM=16
LORA_R=8
LORA_ALPHA=16
NUM_GPUS=8
USE_8BIT=true  # Requires implementation
```

### Implementation Steps

#### Step 1: Update Memory Allocations

Edit `scripts/fine_tune_multi_node.sh`:

```bash
# Line 26-27: Update memory requirements
MEM_FULL="700G"      # For Full FT (7B model)
MEM_LORA="150G"      # For LoRA (7B model)
```

#### Step 2: Adjust Batch Size and Gradient Accumulation

Edit training arguments in the launch command:

```bash
# Reduce batch size, increase gradient accumulation
--per-device-train-batch-size 1 \
--gradient-accumulation-steps 16 \
```

**Effective batch size** = `per_device_batch_size × num_gpus × grad_accum_steps`

#### Step 3: Modify LoRA Configuration

For bigger models, you may want to adjust LoRA parameters in `paramem/fine_tune_with_ckpts.py`:

```python
# Line 102-110: Adjust LoRA config
lora_config = LoraConfig(
    r=16,  # Increase rank for bigger models
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Add more modules
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)
```

#### Step 4: Enable Checkpointing for Memory Efficiency

Add to training arguments:

```python
gradient_checkpointing=True,  # Reduces memory at cost of speed
gradient_checkpointing_kwargs={"use_reentrant": False},
```

#### Step 5: Consider Quantization (Optional)

For very large models, add 8-bit or 4-bit quantization:

```python
# In get_model() function (line 94):
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # or load_in_4bit=True
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)
```

### Example Launch Commands

#### 7B Model with LoRA:

```bash
#!/bin/bash
MODEL="meta-llama/Llama-2-7b-hf"
DATASET="./data/wikidata_pile.csv"
OUTPUT="./models2/Llama-2-7B-lora-pile"
LR="1e-4"

sbatch --job-name=llama7b-lora \
       --partition=alien \
       --qos=alien \
       --nodes=1 \
       --ntasks-per-node=1 \
       --gres=gpu:2 \
       --mem=150GB \
       --time=48:00:00 \
       --wrap="
source ~/.bashrc
conda activate paramem
cd /home/mmahaut/projects/paramem

torchrun --nproc_per_node=2 --nnodes=1 \
  paramem/fine_tune_with_ckpts.py \
  --model-name '$MODEL' \
  --dataset-name '$DATASET' \
  --output-dir '$OUTPUT' \
  --use-lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate $LR \
  --num-train-epochs 3 \
  --save-steps 500 \
  --logging-steps 10 \
  --fsdp
"
```

#### 13B Model with LoRA:

```bash
MODEL="meta-llama/Llama-2-13b-hf"
OUTPUT="./models2/Llama-2-13B-lora-pile"

sbatch --gres=gpu:4 \
       --mem=300GB \
       --time=72:00:00 \
       --wrap="
torchrun --nproc_per_node=4 --nnodes=1 \
  paramem/fine_tune_with_ckpts.py \
  --model-name '$MODEL' \
  --dataset-name '$DATASET' \
  --output-dir '$OUTPUT' \
  --use-lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --num-train-epochs 3 \
  --save-steps 500 \
  --fsdp
"
```

### Monitoring and Debugging

#### Check GPU Memory Usage:

```bash
# During training
watch -n 1 nvidia-smi

# Or in Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

#### Common Issues and Solutions:

1. **OOM (Out of Memory)**
   - Reduce batch size to 1
   - Increase gradient accumulation
   - Enable gradient checkpointing
   - Try LoRA instead of full FT
   - Use 8-bit quantization

2. **Slow Training**
   - Increase batch size if memory allows
   - Reduce gradient accumulation
   - Disable gradient checkpointing
   - Use more GPUs

3. **NCCL Timeout**
   - Increase timeout in `maybe_init_distributed()`
   - Check network connectivity between nodes
   - Verify SLURM environment variables

### Validation Script

Create a quick validation script to test before full training:

```python
# scripts/tests/test_model_loading.py
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model_name = "meta-llama/Llama-2-7b-hf"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Model size: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Test LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f"LoRA memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

### Next Steps

1. **Test with small subset**: Run 100 steps on 1000 samples to verify configuration
2. **Monitor first epoch**: Watch memory and speed closely
3. **Adjust as needed**: Tune batch size and gradient accumulation
4. **Scale gradually**: Start with 7B before attempting 13B/70B

### Additional Resources

- FSDP Documentation: https://pytorch.org/docs/stable/fsdp.html
- LoRA Paper: https://arxiv.org/abs/2106.09685
- DeepSpeed: https://www.deepspeed.ai/
- HuggingFace Trainer: https://huggingface.co/docs/transformers/main_classes/trainer
