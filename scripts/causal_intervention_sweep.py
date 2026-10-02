import torch
import argparse
import pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.causal_intervention import load_sae_from_checkpoint, FeatureSteeringHook

def test_steering_sweep(model_id, layer_name, sae_path, feature_idx, coeffs, prompts_file, out_csv):
    print(f"--- Intervention Test: Steer Feature {feature_idx} Sweep ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    model.eval()

    sae = load_sae_from_checkpoint(sae_path, device)

    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
            
    if target_module is None:
        raise ValueError(f"Could not find layer '{layer_name}' in model.")

    # Read prompts
    prompts = [
        "The movie I watched last night was generally",
        "I was thinking about the plot and it was really",
        "The acting performance felt completely",
        "Overall, the cinematography and direction were",
        "It was one of the most",
        "My general impression of the film is that"
    ]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    # Base Generation Log Probs
    with torch.no_grad():
        out_base = model(**inputs)
        # Probabilities of the very next token prediction
        base_logits = out_base.logits[:, -1, :]
        base_probs = F.softmax(base_logits, dim=-1)
    
    results = []

    for coeff in coeffs:
        print(f"Testing Coefficient: {coeff}")
        hook = FeatureSteeringHook(sae, feature_idx, coeff)
        hook.register(target_module)
        
        # Test Placebo
        hook_placebo = FeatureSteeringHook(sae, feature_idx, coeff, use_placebo=True)
        # We process steered and placebo
        
        with torch.no_grad():
            out_steered = model(**inputs)
            steered_logits = out_steered.logits[:, -1, :]
            steered_probs = F.softmax(steered_logits, dim=-1)
            
            kl_div = F.kl_div(steered_probs.log(), base_probs, reduction='batchmean').item()
            
            base_top_tokens = base_probs.argmax(dim=-1)
            base_top_prob = base_probs.gather(1, base_top_tokens.unsqueeze(-1)).squeeze()
            steered_top_prob = steered_probs.gather(1, base_top_tokens.unsqueeze(-1)).squeeze()
            avg_prob_shift = (steered_top_prob - base_top_prob).mean().item()

        hook.remove()
        
        hook_placebo.register(target_module)
        with torch.no_grad():
            out_placebo = model(**inputs)
            placebo_logits = out_placebo.logits[:, -1, :]
            placebo_probs = F.softmax(placebo_logits, dim=-1)
            kl_div_placebo = F.kl_div(placebo_probs.log(), base_probs, reduction='batchmean').item()
            
            placebo_top_prob = placebo_probs.gather(1, base_top_tokens.unsqueeze(-1)).squeeze()
            avg_prob_shift_placebo = (placebo_top_prob - base_top_prob).mean().item()
        
        hook_placebo.remove()
        
        results.append({
            "Feature_ID": feature_idx,
            "Coefficient": coeff,
            "KL_Divergence": kl_div,
            "KL_Divergence_Placebo": kl_div_placebo,
            "Avg_Prob_Shift_TopToken": avg_prob_shift,
            "Avg_Prob_Shift_Placebo": avg_prob_shift_placebo
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep SAE feature causality via Activation Steering")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m", help="Host model ID")
    parser.add_argument("--layer", type=str, default="gpt_neox.layers.3", help="Host layer mapped by the SAE")
    parser.add_argument("--sae", type=str, required=True, help="Trained SAE checkpoint (.pt)")
    parser.add_argument("--feature", type=int, required=True, help="Index of the domain MP feature to steer")
    parser.add_argument("--coeffs", type=float, nargs="+", default=[-50.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 50.0], help="List of sweep coefficients")
    parser.add_argument("--prompts", type=str, default="", help="Optional path to prompts (using defaults)")
    parser.add_argument("--out", type=str, default="results/H5_sweep_results.csv", help="Output file path")
    
    args = parser.parse_args()
    test_steering_sweep(args.model, args.layer, args.sae, args.feature, args.coeffs, args.prompts, args.out)

