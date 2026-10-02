# Multi-Node Training Setup Documentation

## Overview

This document explains the multi-node distributed training setup for fine-tuning large language models (8B parameters) using PyTorch with DeepSpeed ZeRO-3 across multiple compute nodes connected via InfiniBand.

## Architecture

### Hardware Configuration
- **Nodes**: 2 compute nodes per training job
- **GPUs per node**: 5× NVIDIA A30 GPUs (23.5 GB VRAM each)
- **Total GPUs**: 10 GPUs across 2 nodes
- **Network**: InfiniBand (200 Gb/s) via Mellanox mlx5 HCA
- **CUDA Version**: 12.1.0 (module loaded)
- **Driver Version**: 535.129.03

### Software Stack
- **PyTorch**: Distributed training with NCCL backend
- **DeepSpeed**: ZeRO-3 optimizer for memory-efficient training
- **SLURM**: Job scheduler and resource manager
- **Conda**: Python environment management (paramem environment)

## Multi-Node Communication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         SLURM sbatch                            │
│  - Allocates 2 nodes, 5 GPUs per node                          │
│  - Sets up shared environment variables                         │
│  - Launches run_ft2.sh on master node                          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    run_ft2.sh (Master Node)                     │
│  - Identifies master node: MASTER_ADDR                          │
│  - Generates random port: MASTER_PORT (49152-65535)            │
│  - Launches worker script via srun on ALL nodes                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│           srun --label --export=ALL /bin/bash                   │
│                    run_ft_worker.sh                             │
│  - Executes on EACH node simultaneously                         │
│  - Each node has unique SLURM_NODEID (0, 1, 2, ...)           │
└───────────┬───────────────────────────┬─────────────────────────┘
            │                           │
    ┌───────▼────────┐          ┌──────▼─────────┐
    │   Node 0       │          │    Node 1      │
    │  (Master)      │          │   (Worker)     │
    └───────┬────────┘          └──────┬─────────┘
            │                           │
            ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         torchrun                                │
│  Each node runs: torchrun --nnodes=2 --nproc_per_node=5        │
│                  --node_rank=$SLURM_NODEID                     │
│                  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT     │
│                                                                 │
│  Creates 5 GPU processes per node (10 total):                  │
│  - Node 0: Ranks 0-4 (Local ranks 0-4)                        │
│  - Node 1: Ranks 5-9 (Local ranks 0-4)                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NCCL Communication                           │
│  - Backend: NCCL over InfiniBand                               │
│  - All ranks form single process group (world_size=10)         │
│  - Collective operations: all_reduce, all_gather, broadcast    │
│  - Cross-node communication via IB, intra-node via NVLink      │
└─────────────────────────────────────────────────────────────────┘
```

## NCCL and InfiniBand Configuration

### What is NCCL?
**NCCL** (NVIDIA Collective Communications Library) is a library providing optimized multi-GPU and multi-node communication primitives. It automatically selects the best transport based on hardware:
- **Intra-node**: PCIe, NVLink, NVSwitch
- **Inter-node**: InfiniBand, TCP/IP, RoCE

### What is InfiniBand?
**InfiniBand** is a high-performance networking technology providing:
- **Low latency**: ~1-2 microseconds
- **High bandwidth**: 200 Gb/s in our setup
- **RDMA support**: Direct memory access without CPU involvement
- **Hardware offload**: Message routing handled by HCA (Host Channel Adapter)

### NCCL Environment Variables

```bash
# Enable InfiniBand transport
export NCCL_IB_DISABLE=0                    # 0 = Use IB, 1 = Disable IB

# Network interface configuration
export NCCL_SOCKET_IFNAME=^lo,docker0       # Exclude loopback and docker interfaces

# InfiniBand HCA (Host Channel Adapter)
export NCCL_IB_HCA=mlx5                     # Mellanox ConnectX-5 or newer

# InfiniBand reliability settings
export NCCL_IB_TIMEOUT=22                   # Timeout in milliseconds
export NCCL_IB_RETRY_CNT=7                  # Number of retries before failure

# Debugging and monitoring
export NCCL_DEBUG=INFO                      # Log level: OFF, WARN, INFO, TRACE
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # PyTorch distributed debugging

# Async error handling
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Non-blocking error detection
```

### How NCCL Uses InfiniBand

1. **Initialization**: NCCL detects available IB devices during `ncclCommInitRank()`
2. **Topology Discovery**: Builds network map showing GPU-NIC affinity
3. **Channel Creation**: Establishes IB Queue Pairs (QPs) between ranks
4. **Data Transfer**: 
   - Small messages: Use IB Send/Recv verbs
   - Large messages: Use RDMA Read/Write
5. **Collectives**: Optimized ring/tree algorithms over IB links

### Verification Test

The `test_multinode_nccl.sh` script verifies multi-node setup:

```python
# Creates tensor on each GPU: rank × 1M ones
tensor = torch.ones(1000, 1000, device=device) * rank

# Performs all-reduce (sum across all GPUs)
dist.all_reduce(tensor)

# Verifies result: (0+1+2+3+...+9) × 1M = 45M
assert tensor.sum() == 45_000_000
```

**Test Results** (Job 1844532):
- ✓ 4 ranks across 2 nodes (2 GPUs per node for test)
- ✓ NCCL initialized successfully on all ranks
- ✓ InfiniBand detected: mlx5_0, State: Active, Rate: 200 Gb/s
- ✓ All-reduce operation completed correctly
- ✓ All 4 ranks passed verification

## DeepSpeed ZeRO-3 Configuration

### Why DeepSpeed for Multi-Node?

DeepSpeed ZeRO (Zero Redundancy Optimizer) Stage 3 partitions:
1. **Optimizer states**: Distributed across all GPUs
2. **Gradients**: Distributed across all GPUs
3. **Model parameters**: Distributed across all GPUs

This allows training models larger than single GPU memory by:
- Keeping only 1/N of parameters on each GPU
- Fetching needed parameters on-demand during forward/backward
- Offloading to CPU RAM when not in use

### Configuration File: `ds_config_full.json`

```json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,              // Overlap communication with computation
        "contiguous_gradients": true,      // Reduce memory fragmentation
        "stage3_prefetch_bucket_size": 5e6,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e8,
        "offload_param": {
            "device": "cpu",               // Offload parameters to CPU RAM
            "pin_memory": true             // Use pinned memory for faster transfer
        }
    },
    "bf16": {"enabled": true},             // Use bfloat16 for reduced memory
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,         // Offload activations to CPU
        "number_checkpoints": 4
    }
}
```

### Memory Distribution (10 GPUs, 8B model)

Without ZeRO-3:
- Each GPU: ~16 GB model + gradients + optimizer = **21+ GB** → OOM!

With ZeRO-3 + CPU offload:
- Each GPU: ~1.6 GB parameters + activations = **~8-10 GB** → Fits!
- CPU RAM: ~140 GB shared across 10 GPUs for offloaded parameters

## Training Script Structure

### 1. Main Launcher: `fine_tune_multi_node.sh`

```bash
# Calculate multi-node distribution
NUM_NODES=2
GPUS_PER_NODE=5
GRES="gpu:5"

# Launch SLURM job
sbatch \
  --nodes=$NUM_NODES \
  --gres=$GRES \
  --ntasks-per-node=1 \          # 1 task per node (runs torchrun)
  --cpus-per-task=4 \
  --mem=180G \                    # 180GB RAM per node for full FT
  --partition=alien \
  --exclude=node044 \             # Exclude problematic nodes
  run_ft2.sh
```

### 2. Master Coordinator: `run_ft2.sh`

```bash
# Set up rendezvous for multi-node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$(( ( RANDOM % 16383 ) + 49152 ))

# Launch worker script on ALL nodes via srun
srun --label --export=ALL /bin/bash run_ft_worker.sh
```

**Key Points**:
- `srun` launches the script on EACH allocated node
- `--export=ALL` passes all environment variables to workers
- `--label` prefixes output with node ID for debugging

### 3. Worker Script: `run_ft_worker.sh`

```bash
# Activate environment on each node
source ~/.bashrc
module load CUDA/12.1.0
conda activate paramem

# Set NCCL configuration
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_SOCKET_IFNAME=^lo,docker0

# Launch torchrun on this node
torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \           # Total nodes in job
  --nproc_per_node=$GPUS_PER_NODE \         # GPUs to use on this node
  --node_rank=$SLURM_NODEID \               # This node's ID (0, 1, ...)
  --rdzv_id=$SLURM_JOB_ID \                 # Unique ID for this job
  --rdzv_backend=c10d \                      # Rendezvous backend
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \ # Master node address
  fine_tune_with_ckpts.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --deep-speed \                             # Use DeepSpeed instead of FSDP
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  ...
```

**Key Points**:
- Each node runs this script independently
- `torchrun` creates GPU processes (world_size = nnodes × nproc_per_node)
- All processes connect to master via `rdzv_endpoint`
- SLURM provides unique `SLURM_NODEID` for each node

### 4. Training Script: `fine_tune_with_ckpts.py`

```python
import torch.distributed as dist
from transformers import TrainingArguments
from deepspeed import DeepSpeedEngine

# PyTorch distributed is initialized by torchrun automatically
# via environment variables: RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

training_args = TrainingArguments(
    output_dir="./checkpoint",
    deepspeed="ds_config_full.json",        # DeepSpeed config
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,          # Effective batch = 1 × 8 × 10 = 80
    bf16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    ...
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()  # Training happens across all 10 GPUs automatically
```

## Troubleshooting Common Issues

### Issue 1: NCCL Timeout / Communication Failure

**Symptoms**:
```
[E ProcessGroupNCCL.cpp:1785] Exception (either an error or timeout) detected by watchdog
```

**Causes**:
- InfiniBand not enabled (`NCCL_IB_DISABLE=1`)
- Wrong network interface selected
- Firewall blocking communication
- Incorrect MASTER_ADDR or MASTER_PORT

**Solution**:
```bash
# Enable InfiniBand
export NCCL_IB_DISABLE=0

# Verify IB is active
ibstat  # Should show "State: Active"

# Test with NCCL test
sbatch test_multinode_nccl.sh
```

### Issue 2: Worker Script Not Found on Secondary Nodes

**Symptoms**:
```
srun: error: node038: task 1: No such file or directory
```

**Cause**: Script created in `/tmp/` on master node only, not accessible from other nodes

**Solution**: Use shared storage paths:
```bash
# ✗ Wrong - only exists on master node
cat > /tmp/worker.sh << 'EOF'
...
EOF
srun /tmp/worker.sh

# ✓ Correct - accessible from all nodes
cat > /home/user/shared/worker.sh << 'EOF'
...
EOF
srun /home/user/shared/worker.sh
```

### Issue 3: Conda Environment Not Activated in srun

**Symptoms**:
```
srun: error: torchrun: command not found
```

**Cause**: `srun` creates fresh shell without sourcing `.bashrc`

**Solution**: Explicitly source in worker script:
```bash
#!/bin/bash
source ~/.bashrc
conda activate paramem
# Now conda commands work
```

### Issue 4: CUDA Out of Memory Despite Multi-Node

**Symptoms**:
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 14.96 GiB
```

**Cause**: Model not properly sharded or CPU offload not working

**Solution**:
1. Verify DeepSpeed config loaded: Check logs for "DeepSpeed ZeRO Stage 3"
2. Enable parameter offload: `"offload_param": {"device": "cpu"}`
3. Increase activation checkpointing: `"number_checkpoints": 4`
4. Reduce batch size: `per_device_train_batch_size=1`

### Issue 5: Different CUDA Versions Across Nodes

**Symptoms**:
```
NCCL version mismatch
RuntimeError: CUDA error: invalid device function
```

**Cause**: NCCL compiled against different CUDA version than runtime

**Solution**:
```bash
# Load same CUDA module on all nodes
module load CUDA/12.1.0

# Verify on all nodes
srun bash -c "module load CUDA/12.1.0 && nvcc --version"
```

## Performance Optimization

### Network Bandwidth Utilization

Monitor InfiniBand usage during training:
```bash
# On each node, check IB counters
watch -n 1 'ibstat mlx5_0 | grep -A 20 "Port 1"'

# Look for:
# - PortXmitData: Data transmitted
# - PortRcvData: Data received
# Should increase steadily during training
```

### GPU Utilization

```bash
# Monitor GPU usage across all nodes
srun nvidia-smi dmon -s u

# During training should show:
# - GPU Utilization: 80-95%
# - Memory Usage: 8-12 GB (with ZeRO-3 + offload)
```

### Gradient Accumulation Tuning

Effective batch size = `batch_size × grad_accum × world_size`

For our setup:
- batch_size = 1
- grad_accum = 8
- world_size = 10 GPUs
- **Effective batch = 80 samples**

Increase grad_accum if:
- Need larger effective batch for stability
- Have extra memory headroom

### Communication Overlap

DeepSpeed overlaps communication with computation:
```json
{
    "zero_optimization": {
        "overlap_comm": true,              // Communicate while computing
        "contiguous_gradients": true,      // Reduce fragmentation
        "stage3_prefetch_bucket_size": 5e6 // Prefetch next layer params
    }
}
```

## Monitoring Training Progress

### Check Job Status
```bash
# View running jobs
squeue -u $USER

# Check job details
sacct -j <job_id> --format=JobID,State,Elapsed,NodeList

# View live output
tail -f models2/model-name/slurm_logs/<job_id>_node000.out
```

### Training Metrics

Look for in logs:
```
DeepSpeed info: version=0.14.5
world_size = 10
train_batch_size = 80
ZeRO stage 3 optimizer
offload_param = cpu

Step 1/1540 | Loss: 2.456 | LR: 1e-4
Step 10/1540 | Loss: 2.234 | LR: 1e-4
...
```

### Checkpoint Saving

With multi-node training, DeepSpeed saves:
```
checkpoint-100/
├── global_step100/
│   ├── zero_pp_rank_0_mp_rank_00_model_states.pt
│   ├── zero_pp_rank_1_mp_rank_00_model_states.pt
│   ├── ...
│   └── zero_pp_rank_9_mp_rank_00_model_states.pt
└── mp_rank_00_model_states.pt  # Consolidated checkpoint
```

## Key Takeaways

1. **Multi-node training requires**: SLURM → srun → torchrun → NCCL
2. **InfiniBand is critical**: 10× faster than Ethernet for inter-node communication
3. **NCCL auto-detects IB**: But needs correct environment variables
4. **DeepSpeed ZeRO-3 enables large models**: By partitioning parameters across GPUs
5. **CPU offload saves GPU memory**: At cost of some PCIe bandwidth
6. **Shared storage is essential**: Worker scripts must be accessible from all nodes
7. **Testing is crucial**: Use `test_multinode_nccl.sh` before full training

## References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [InfiniBand Architecture](https://www.openfabrics.org/)
- [SLURM Multi-Node Jobs](https://slurm.schedmd.com/multi_cluster.html)
