import torch
import torch.nn as nn
from src.sae.model import SparseAutoencoder
from src.sae.mp_utils import compute_dataset_mp_threshold, identify_domain_features

def main():
    print("Initialize Step 1: Model and Data Setup")
    
    # Mock parameters representing a typical LLM layer output
    d_model = 768   # Pythia / GPT-2 small typical dimension
    expansion_factor = 4
    dict_size = d_model * expansion_factor
    
    # Initialize the SAE model
    sae = SparseAutoencoder(d_model=d_model, dict_size=dict_size)
    sae.eval() # Evaluating pre-trained features
    
    # Generate Synthetic Activations mimicking General and Domain Data
    # In reality, this would be `activations = model(tokens).hidden_states`
    print("Step 2: Collect Feature Activations")
    N_tokens = 5000 
    
    # General Data: Activations are mostly noise with some sparse features
    # features 0-10 are universal, firing randomly
    general_acts = torch.randn(N_tokens, d_model)
    
    # Domain Data: Has specific directional shifts
    domain_acts = torch.randn(N_tokens, d_model)
    domain_acts[:, 0] += 5.0 # Artificially inject a dominant domain feature direction
    
    with torch.no_grad():
        print("Extracting SAE features (Forward Pass)...")
        _, _, f_general, _, _ = sae(general_acts)
        _, _, f_domain, _, _ = sae(domain_acts)
        
    print("Step 3: Calculating Marchenko-Pastur bulk thresholds")
    gen_mask, gen_lmax, gen_sig, gen_var = compute_dataset_mp_threshold(f_general)
    dom_mask, dom_lmax, dom_sig, dom_var = compute_dataset_mp_threshold(f_domain)
    
    print(f"General MP Edge (lambda_max): {gen_lmax:.4f} (Sigma^2 est: {gen_sig:.4f})")
    print(f"Domain MP Edge (lambda_max): {dom_lmax:.4f} (Sigma^2 est: {dom_sig:.4f})")
    
    print("Step 4: Mapping Domain specific features")
    feature_categories = identify_domain_features(gen_mask, dom_mask)
    
    num_universal = len(feature_categories['universal'])
    num_domain = len(feature_categories['domain'])
    num_dead = len(feature_categories['dead'])
    
    print("\nResults Summary (Number of Features):")
    print("-" * 40)
    print(f"Universal Features: {num_universal}")
    print(f"Domain Specific Features: {num_domain}")
    print(f"Dead / Substrate Features: {num_dead}")
    
    print("\nNext Steps: Pipeline to Auto-Annotator & Causal Intervention")
    # Compare with Activation Energy standard thresholding:
    std_energy_threshold = f_domain.mean(dim=0) + 2 * f_domain.std(dim=0)
    energy_mask = dom_var > std_energy_threshold.var()
    print(f"\nFeatures selected by arbitrary Activation Energy (+2 std): {energy_mask.sum()}")

if __name__ == "__main__":
    main()
