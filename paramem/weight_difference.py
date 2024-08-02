# load a base and a fine-tune model and compare the hidden states
# identify at which layer the hidden states differ the most
# imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import sys
import typer
import logging
from paramem.load_distrib_model import load_distrib_model
import seaborn as sns
import matplotlib.pyplot as plt

# load models
def load_model(model_name, checkpoint_path=None):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if checkpoint_path:
        try:
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model checkpoint: {e} Trying to load as distributed model.")
            model = load_distrib_model(model, checkpoint_path)
    return model, tokenizer

def main():
    # load models
    base_model, base_tokenizer = load_model("meta-llama/Meta-Llama-3-8B")
    fine_tuned_model, fine_tuned_tokenizer = load_model("meta-llama/Meta-Llama-3-8B", "./models/Met7-ft/checkpoint-2000/pytorch_model_fsdp_0")
    weight_types = [
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight"
    ]
    # compare weight matrices
    # Lama3
    ft_effects={w_type: [] for w_type in weight_types}
    for layer_id in range(len(base_model.model.layers)):
        for weight_type in weight_types:
            ft_module=None
            base_module=None
            for attr in weight_type.split("."):
                if ft_module is None:
                    ft_module = fine_tuned_model.model.layers[layer_id].__getattr__(attr)
                    base_module = base_model.model.layers[layer_id].__getattr__(attr)
                else:
                    ft_module = ft_module.__getattr__(attr)
                    base_module = base_module.__getattr__(attr)
            # diff = torch.norm(base_module-ft_module).item() # use a dot product here
            diff = torch.dot(base_module.flatten(),ft_module.flatten()).item()
            ft_effects[weight_type].append(diff)
    
    # plot differences
    for weight_type, diffs in ft_effects.items():
        sns.lineplot(x=range(len(diffs)), y=diffs, label=weight_type)
    plt.legend()
    plt.savefig("weight_diffs.png")

if __name__ == "__main__":
    main()