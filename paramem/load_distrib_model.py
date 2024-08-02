import torch
from transformers import AutoModelForCausalLM
# from torch.distributed._shard.state_dict import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.checkpoint import FileSystemReader
from accelerate.utils.fsdp_utils import merge_fsdp_weights
from pathlib import Path

def load_a_checkpoint(model, distcp_checkpoint_path: str):
    """
    Load a checkpoint from fsdp .distcp and return the state_dict
    """
    state_dict = {
            "model": model.state_dict()
        }
    dist_cp.load_state_dict(
                    state_dict=state_dict,
                    storage_reader= FileSystemReader(distcp_checkpoint_path),
                    no_dist=True,
                )
    return state_dict["model"], state_dict["optimizer"]
    
def load_distrib_model(model, path):
    print(f"preparing distributed model from {path} for {model.__class__.__name__}")
    output_dir = Path(path).parent
    merge_fsdp_weights(path, output_dir, safe_serialization=False)
    state_dict = torch.load(output_dir / "pytorch_model.bin")
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")
    model = load_distrib_model(model, "/home/mmahaut/projects/paramem/models/Mis7-ft/checkpoint-1000/pytorch_model_fsdp_0")
    print("Model loaded successfully")