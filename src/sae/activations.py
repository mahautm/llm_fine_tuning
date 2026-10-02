import torch
import torch.nn as nn
from typing import Dict, List, Optional
from tqdm import tqdm

class ActivationExtractor:
    """Helper class to extract activations from designated layers of a PyTorch model."""
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {name: [] for name in layer_names}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Attach forward hooks to specified layers."""
        def get_hook(name):
            def hook(module, input, output):
                # Detach and move to CPU immediately to save GPU memory
                # If output is a tuple (common in HF), grab the first element (hidden states)
                if isinstance(output, tuple):
                    acts = output[0].detach().cpu()
                else:
                    acts = output.detach().cpu()
                self.activations[name].append(acts)
            return hook

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(get_hook(name)))
                
    def extract(self, dataloader, num_batches: int = 10, target_layer: str = None) -> torch.Tensor:
        """Run dataloader up to num_batches and return concatenated activations."""
        self.model.eval()
        # Reset storage
        self.activations = {name: [] for name in self.layer_names}
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Extracting Activations", total=num_batches)):
                if i >= num_batches:
                    break
                    
                # Move batch to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass - the hooks will catch the activations
                try:
                    self.model(**batch)
                except Exception as e:
                    # In case of sequence length issues or other batch errors
                    print(f"Batch inference failed: {e}")
                    continue
                    
        # Concatenate gathered activations across all batches
        # [num_batches * batch_size, sequence_length, hidden_dim]
        target = target_layer if target_layer else self.layer_names[0]
        
        if not self.activations[target]:
            raise ValueError(f"No activations found for layer {target}. Did the hook attached properly?")
            
        all_acts = torch.cat(self.activations[target], dim=0)
        
        # Flatten num_batches and sequence_length -> [total_tokens, hidden_dim]
        # Text/Images represent different token shapes, flatten everything except hidden_dim
        dim = all_acts.shape[-1]
        return all_acts.view(-1, dim)
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
