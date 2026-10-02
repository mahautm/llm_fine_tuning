import torch
import argparse
import pandas as pd
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from PIL import Image

from scripts.causal_intervention import load_sae_from_checkpoint, FeatureSteeringHook

def test_steering_vision_sweep(model_id, layer_name, sae_path, feature_idx, coeffs, dataset_name, out_csv):
    print(f"--- Intervention Test: Steer Vision Feature {feature_idx} Sweep ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    model.eval()

    sae = load_sae_from_checkpoint(sae_path, device)

    # Locate target module mapping. Usually something like 'vit.encoder.layer.6' => we need to resolve nested objects.
    # We recursively find it. 
    target_module = None
    for name, module in model.named_modules():
        if name.endswith(layer_name):
            target_module = module
            break
            
    if target_module is None:
        # Fallback to the explicit vit encoder mapping structure commonly seen in huggingface
        target_module = model.vit.encoder.layer[int(layer_name.split('.')[-1])]

    # Get sample images
    dataset = load_dataset(dataset_name, split="train")
    img_column = 'image' if 'image' in dataset.column_names else 'img'
    
    # We sample a diverse set of class indices
    samples = []
    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), 16)
    
    images = [dataset[i][img_column].convert('RGB') for i in indices]
    inputs = processor(images=images, return_tensors="pt").to(device)
    
    # Base classification probabilities
    with torch.no_grad():
        out_base = model(**inputs)
        base_logits = out_base.logits
        base_probs = F.softmax(base_logits, dim=-1)
    
    results = []

    for coeff in coeffs:
        print(f"Testing Steer Coefficient: {coeff}")
        hook = FeatureSteeringHook(sae, feature_idx, coeff)
        hook.register(target_module)
        
        with torch.no_grad():
            out_steered = model(**inputs)
            steered_logits = out_steered.logits
            steered_probs = F.softmax(steered_logits, dim=-1)
            
            kl_div = F.kl_div(steered_probs.log(), base_probs, reduction='batchmean').item()
            
            base_top_classes = base_probs.argmax(dim=-1)
            base_top_prob = base_probs.gather(1, base_top_classes.unsqueeze(-1)).squeeze(-1)
            steered_top_prob = steered_probs.gather(1, base_top_classes.unsqueeze(-1)).squeeze(-1)
            avg_prob_shift = (steered_top_prob - base_top_prob).mean().item()
            
            steered_top_classes = steered_probs.argmax(dim=-1)
            label_shifts = (base_top_classes != steered_top_classes).sum().item() / len(indices)

        hook.remove()

        hook_placebo = FeatureSteeringHook(sae, feature_idx, coeff, use_placebo=True)
        hook_placebo.register(target_module)
        
        with torch.no_grad():
            out_placebo = model(**inputs)
            placebo_logits = out_placebo.logits
            placebo_probs = F.softmax(placebo_logits, dim=-1)
            
            kl_div_placebo = F.kl_div(placebo_probs.log(), base_probs, reduction='batchmean').item()
            
            placebo_top_prob = placebo_probs.gather(1, base_top_classes.unsqueeze(-1)).squeeze(-1)
            avg_prob_shift_placebo = (placebo_top_prob - base_top_prob).mean().item()
            
            placebo_top_classes = placebo_probs.argmax(dim=-1)
            label_shifts_placebo = (base_top_classes != placebo_top_classes).sum().item() / len(indices)

        hook_placebo.remove()
        
        results.append({
            "Feature_ID": feature_idx,
            "Coefficient": coeff,
            "KL_Divergence": kl_div,
            "KL_Divergence_Placebo": kl_div_placebo,
            "Avg_Prob_Shift_TopClass": avg_prob_shift,
            "Avg_Prob_Shift_Placebo": avg_prob_shift_placebo,
            "Label_Flip_Ratio": label_shifts,
            "Label_Flip_Ratio_Placebo": label_shifts_placebo
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved vision intervention sweep results to {out_csv}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224", help="Host model ID")
    parser.add_argument("--layer", type=str, default="encoder.layer.6", help="Host layer mapped by the SAE")
    parser.add_argument("--sae", type=str, required=True, help="Trained SAE checkpoint (.pt)")
    parser.add_argument("--feature", type=int, required=True, help="Index of the domain MP feature to steer")
    parser.add_argument("--coeffs", type=float, nargs="+", default=[-100.0, -50.0, -20.0, 0.0, 20.0, 50.0, 100.0], help="Coefficients")
    parser.add_argument("--dataset", type=str, default="cifar10", help="HuggingFace dataset name")
    parser.add_argument("--out", type=str, default="results/V5_vision_sweep.csv", help="Output file path")
    args = parser.parse_args()
    
    test_steering_vision_sweep(args.model, args.layer, args.sae, args.feature, args.coeffs, args.dataset, args.out)
