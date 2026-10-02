#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --time=00:10:00
#SBATCH --job-name=nccl_test
#SBATCH --output=nccl_test_%j_%N.out
#SBATCH --error=nccl_test_%j_%N.err

source ~/.bashrc
module load CUDA/12.1.0
conda activate paramem

# Set up multi-node communication
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$(( ( RANDOM % 16383 ) + 49152 ))

echo "=== NCCL Multi-Node Test ==="
echo "Node: $(hostname)"
echo "Node ID: $SLURM_NODEID"
echo "Nodes: $SLURM_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "CUDA Version:"
nvcc --version || echo "nvcc not found"
nvidia-smi | grep "CUDA Version"
echo ""
echo "InfiniBand Status:"
ibstat || echo "ibstat not available"
echo ""
echo "NCCL Environment:"
env | grep -E "NCCL|CUDA|IB" | sort

# Enhanced NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Create a simple PyTorch NCCL test
cat > /tmp/nccl_test_$SLURM_JOB_ID.py << 'PYEOF'
import torch
import torch.distributed as dist
import os
import socket

def test_nccl():
    print(f"[{socket.gethostname()}] Starting NCCL test")
    print(f"[{socket.gethostname()}] CUDA available: {torch.cuda.is_available()}")
    print(f"[{socket.gethostname()}] CUDA device count: {torch.cuda.device_count()}")
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"[{socket.gethostname()}] Rank {rank}/{world_size}, Local rank {local_rank}")
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Create tensor and do all-reduce
    tensor = torch.ones(1000, 1000, device=device) * rank
    print(f"[{socket.gethostname()}] Rank {rank}: tensor sum before all-reduce = {tensor.sum().item()}")
    
    dist.all_reduce(tensor)
    expected = sum(range(world_size)) * 1000000
    print(f"[{socket.gethostname()}] Rank {rank}: tensor sum after all-reduce = {tensor.sum().item()} (expected: {expected})")
    
    # Verify
    if abs(tensor.sum().item() - expected) < 1:
        print(f"[{socket.gethostname()}] Rank {rank}: ✓ NCCL test PASSED")
    else:
        print(f"[{socket.gethostname()}] Rank {rank}: ✗ NCCL test FAILED")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    test_nccl()
PYEOF

echo ""
echo "=== Running PyTorch NCCL Test ==="
# Copy test script to shared location accessible from all nodes
cp /tmp/nccl_test_$SLURM_JOB_ID.py /home/mmahaut/projects/paramem/scripts/nccl_test_$SLURM_JOB_ID.py

# Create a worker script in shared storage
cat > /home/mmahaut/projects/paramem/scripts/nccl_worker_$SLURM_JOB_ID.sh << 'WORKER'
#!/bin/bash
source ~/.bashrc
module load CUDA/12.1.0
conda activate paramem

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=2 \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /home/mmahaut/projects/paramem/scripts/nccl_test_$SLURM_JOB_ID.py
WORKER
chmod +x /home/mmahaut/projects/paramem/scripts/nccl_worker_$SLURM_JOB_ID.sh

srun --label --export=ALL /bin/bash /home/mmahaut/projects/paramem/scripts/nccl_worker_$SLURM_JOB_ID.sh

echo ""
echo "=== Test Complete ==="
