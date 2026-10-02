import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_sae_from_checkpoint(sae_path, device="cuda"):
    # Mocking standard imports to preserve independence of script
    from src.sae.model import SparseAutoencoder
    from src.sae.topk_model import TopKAutoencoder, BatchTopKAutoencoder
    
    checkpoint = torch.load(sae_path, map_location=device)
    sae_type = checkpoint.get('sae_type', 'standard')
    
    if sae_type == 'standard':
        sae = SparseAutoencoder(
            d_model=checkpoint['d_model'], 
            dict_size=checkpoint['dict_size'], 
            l1_coeff=checkpoint.get('l1_coeff', 0.0)
        )
    elif sae_type == 'topk':
        sae = TopKAutoencoder(
            d_model=checkpoint['d_model'], 
            dict_size=checkpoint['dict_size'], 
            k=checkpoint.get('top_k', 32)
        )
    elif sae_type == 'batch_topk':
        sae = BatchTopKAutoencoder(
            d_model=checkpoint['d_model'], 
            dict_size=checkpoint['dict_size'], 
            k_per_batch=checkpoint.get('top_k', 32) * 8192
        )
        
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval().to(device)
    return sae


class FeatureSteeringHook:
    """Intervenes in the forward pass of an LM by steering a specific SAE feature or a random direction."""
    def __init__(self, sae, feature_idx, steering_coeff, use_placebo=False):
        self.sae = sae
        self.feature_idx = feature_idx
        self.steering_coeff = steering_coeff
        self.use_placebo = use_placebo
        self.handle = None
        self.placebo_vector = None

    def hook_fn(self, module, inputs, outputs):
        is_tuple = isinstance(outputs, tuple)
        hidden_states = outputs[0] if is_tuple else outputs
        
        with torch.no_grad():
            if self.use_placebo:
                # Generate a random orthonormal vector matching the decoder weight norm
                if self.placebo_vector is None:
                    rand_dir = torch.randn(hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
                    rand_dir = rand_dir / torch.norm(rand_dir)
                    norm_factor = torch.norm(self.sae.decoder.weight[:, self.feature_idx] if hasattr(self.sae, 'decoder') else self.sae.W_dec[self.feature_idx])
                    self.placebo_vector = rand_dir * norm_factor
                
                delta = (self.placebo_vector * self.steering_coeff).to(hidden_states.dtype)
                steered_hidden_states = hidden_states + delta
            else:
                f_orig = self.sae.encode(hidden_states)
                x_recon_orig = self.sae.decode(f_orig)
                
                f_steered = f_orig.clone()
                f_steered[..., self.feature_idx] += self.steering_coeff
                x_recon_steered = self.sae.decode(f_steered)

                delta = (x_recon_steered - x_recon_orig).to(hidden_states.dtype)
                steered_hidden_states = hidden_states + delta
            
        if is_tuple:
            return (steered_hidden_states,) + outputs[1:]
        return steered_hidden_states

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def test_steering(model_id, layer_name, sae_path, feature_idx, steering_coeff, prompt="The movie I watched last night was generally"):
    print(f"--- Intervention Test: Steer Feature {feature_idx} by {steering_coeff} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Host Model
    print(f"Loading Host LM: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    model.eval()

    # 2. Load SAE
    print(f"Loading SAE: {sae_path}")
    sae = load_sae_from_checkpoint(sae_path, device)

    # 3. Locate Target Layer to hook
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
            
    if target_module is None:
        raise ValueError(f"Could not find layer '{layer_name}' in model.")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Base Generation (No Steering)
    with torch.no_grad():
        out_base = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
    print("\n[Baseline Output]:")
    print(tokenizer.decode(out_base[0], skip_special_tokens=True))

    # Steered Generation
    hook = FeatureSteeringHook(sae, feature_idx, steering_coeff)
    hook.register(target_module)
    
    # Re-run and seed with same manual state if doing strict science, here we just observe the qualitative shift
    with torch.no_grad():
        out_steered = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.eos_token_id)
        
    hook.remove()

    print(f"\n[Steered Output (Concept injected @ {steering_coeff})]:")
    print(tokenizer.decode(out_steered[0], skip_special_tokens=True))
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SAE feature causality via Activation Steering")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m", help="Host model ID")
    parser.add_argument("--layer", type=str, default="gpt_neox.layers.3", help="Host layer mapped by the SAE")
    parser.add_argument("--sae", type=str, required=True, help="Trained SAE checkpoint (.pt)")
    parser.add_argument("--feature", type=int, required=True, help="Index of the domain MP feature to steer")
    parser.add_argument("--coeff", type=float, default=20.0, help="Strength of the steering coefficient to add")
    parser.add_argument("--prompt", type=str, default="The robot navigated the field to", help="Test prompt")
    
    args = parser.parse_args()
    test_steering(args.model, args.layer, args.sae, args.feature, args.coeff, args.prompt)
