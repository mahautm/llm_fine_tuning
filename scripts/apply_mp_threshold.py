import torch
import argparse
import os
import json
from src.sae.mp_utils import compute_dataset_mp_threshold, identify_domain_features

def threshold_and_map(
    f_general_path: str,
    f_domain_path: str,
    output_report: str,
    scale_mode: str,
    clip_quantile: float | None,
    sigma_estimator: str,
    sigma_quantile: float,
    bottom_fraction: float,
):
    """Computes MP threshold on general and domain SAE features, then categorizes them."""
    print("Loading general feature activations...")
    f_gen = torch.load(f_general_path)
    
    print("Loading domain feature activations...")
    f_dom = torch.load(f_domain_path)

    # 1. MP thresholds
    print("Computing MP bounds...")
    gen_mask, gen_lmax, gen_sig, gen_var = compute_dataset_mp_threshold(
        f_gen.float(),
        scale_mode=scale_mode,
        clip_quantile=clip_quantile,
        sigma_estimator=sigma_estimator,
        sigma_quantile=sigma_quantile,
        bottom_fraction=bottom_fraction,
    )
    dom_mask, dom_lmax, dom_sig, dom_var = compute_dataset_mp_threshold(
        f_dom.float(),
        scale_mode=scale_mode,
        clip_quantile=clip_quantile,
        sigma_estimator=sigma_estimator,
        sigma_quantile=sigma_quantile,
        bottom_fraction=bottom_fraction,
    )
    
    # 2. Map Features
    categories = identify_domain_features(gen_mask, dom_mask)
    
    # Optional baseline 1: activation energy > 0 (Dead/Alive) standard L1
    # Often, features are dead if they just don't fire. 
    # But some might merely 'flicker' in the noise without breaking MP.
    is_alive_gen = (f_gen > 0).sum(dim=0) > 10  # fires more than 10 times
    is_alive_dom = (f_dom > 0).sum(dim=0) > 10
    
    # 3. Export
    report = {
        "MP_Config": {
            "scale_mode": scale_mode,
            "clip_quantile": None if clip_quantile is None else float(clip_quantile),
            "sigma_estimator": sigma_estimator,
            "sigma_quantile": float(sigma_quantile),
            "bottom_fraction": float(bottom_fraction),
        },
        "MP_Thresholds": {
            "General_lambda_max": float(gen_lmax),
            "General_sigma_sq": float(gen_sig),
            "Domain_lambda_max": float(dom_lmax),
            "Domain_sigma_sq": float(dom_sig)
        },
        "Feature_Counts_MP": {
            "Universal": int(len(categories['universal'])),
            "Domain_Specific": int(len(categories['domain'])),
            "Dead_or_Substrate": int(len(categories['dead']))
        },
        "Feature_Counts_Basic_Energy": {
            "Only_Active_In_Domain": int(((~is_alive_gen) & is_alive_dom).sum().item()),
            "Active_In_Both": int((is_alive_gen & is_alive_dom).sum().item()),
            "Dead": int(((~is_alive_gen) & (~is_alive_dom)).sum().item())
        },
        "Categories": {
            k: v.tolist() for k, v in categories.items()
        }
    }
    
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n--- MP Thresholding Results ---")
    print(f"Universal: {report['Feature_Counts_MP']['Universal']}")
    print(f"Domain:    {report['Feature_Counts_MP']['Domain_Specific']} (These are your new monosemantic candidates)")
    print(f"Dead/Noise:{report['Feature_Counts_MP']['Dead_or_Substrate']}")
    print(f"Report saved to {output_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_gen", type=str, required=True, help="Path to General dataset SAE features (.pt)")
    parser.add_argument("--f_dom", type=str, required=True, help="Path to Domain dataset SAE features (.pt)")
    parser.add_argument("--out_json", type=str, required=True, help="Output JSON report file")
    parser.add_argument("--scale_mode", type=str, default="none", choices=["none", "zscore", "mad"], help="Feature-wise robust scaling mode")
    parser.add_argument("--clip_quantile", type=float, default=None, help="Optional global abs-value clipping quantile in [0.5, 0.9999]")
    parser.add_argument("--sigma_estimator", type=str, default="bottom_mean", choices=["bottom_mean", "median", "quantile"], help="Noise variance estimator from eigen spectrum")
    parser.add_argument("--sigma_quantile", type=float, default=0.5, help="Quantile for sigma estimator when --sigma_estimator=quantile")
    parser.add_argument("--bottom_fraction", type=float, default=0.8, help="Bottom-bulk fraction for --sigma_estimator=bottom_mean")
    
    args = parser.parse_args()
    threshold_and_map(
        args.f_gen,
        args.f_dom,
        args.out_json,
        scale_mode=args.scale_mode,
        clip_quantile=args.clip_quantile,
        sigma_estimator=args.sigma_estimator,
        sigma_quantile=args.sigma_quantile,
        bottom_fraction=args.bottom_fraction,
    )
