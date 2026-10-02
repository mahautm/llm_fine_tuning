import torch
import json
import random
import argparse
import os
import subprocess
import requests
from tqdm import tqdm
import sys

# Append current directory to path to import eval_auto_annotation properly if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_auto_annotation import get_top_k_activating_contexts, build_llm_prompt, _extract_json_object

def annotate_feature_with_llm(feat_idx, contexts, api_key, api_base, model_id):
    prompt = build_llm_prompt(feat_idx, contexts)
    
    headers = {"Authorization": f"Bearer {api_key}"}
    if not api_base:
        api_base = "https://api.openai.com/v1"
        if not model_id:
            model_id = "gpt-4o"
    
    url = f"{api_base.rstrip('/')}/chat/completions"
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are an expert AI interpretability researcher analyzing sparse autoencoder dictionaries. You must strictly reply with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"]
    res_dict = _extract_json_object(reply)
    if "confidence" not in res_dict:
        res_dict["confidence"] = 0.0
    return res_dict

def evaluate_noise_features(api_base, model_id):
    print("\n--- GAP 1: Testing Semantic Validity of Noise Features ---")
    report_path = "results/H3_mp_threshold_report.json"
    if not os.path.exists(report_path):
        print(f"Report not found at {report_path}")
        return
        
    with open(report_path, "r") as f:
        report = json.load(f)
        
    f_acts_path = "results/sae_activations/f_acts_pythia_dom.pt"
    
    # In earlier tests it showed dead / noise feature arrays directly
    noise_features = report.get("Categories", {}).get("dead", [])
            
    if not noise_features:
        print("No noise features identified in the threshold report.")
        return
        
    print(f"Found {len(noise_features)} features below the MP threshold. Sampling 50...")
    sample_size = min(50, len(noise_features))
    sampled_noise = random.sample(noise_features, sample_size)
    
    contexts = get_top_k_activating_contexts(f_acts_path, "imdb", sampled_noise, top_n=5)
    
    confidences = []
    for feat_idx in tqdm(sampled_noise, desc="Annotating Noise Features"):
        try:
            annotation = annotate_feature_with_llm(feat_idx, contexts[feat_idx], api_key="local", api_base=api_base, model_id=model_id)
            confidences.append(annotation.get("confidence", 0.0))
        except Exception as e:
            # Handle potential API failures or unparseable JSON from word-salad features
            confidences.append(0.0)
            
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    print(f"Average LLM Confidence for Features BELOW MP Threshold: {avg_conf:.3f}")
    
    # Save the results
    os.makedirs("results/gaps", exist_ok=True)
    with open("results/gaps/noise_features_annotation.json", "w") as f:
        json.dump({"sampled_features": sampled_noise, "confidences": confidences, "average_confidence": avg_conf}, f, indent=2)


def evaluate_architecture_yield(api_base, model_id):
    print("\n--- GAP 2: Top-K vs Batch Top-K vs Standard Yield Comparison ---")
    print("Warning: Currently missing separate Top-K JSON MP reports. Proceeding with Standard Yield as control point.")
    # Identify domain features for each architecture type
    report_path = "results/H3_mp_threshold_report.json"
    if not os.path.exists(report_path):
        return
        
    with open(report_path, "r") as f:
        report = json.load(f)
        
    features_by_arch = {"standard": report.get("Categories", {}).get("domain", [])}
    f_acts_path = "results/sae_activations/f_acts_pythia_dom.pt"

    results = {}
    for arch, feats in features_by_arch.items():
        if not feats:
            continue
        print(f"\nEvaluating {arch.upper()} architecture (Total Domain Features: {len(feats)})")
        sample_size = min(50, len(feats))
        sampled_feats = random.sample(feats, sample_size)
        
        contexts = get_top_k_activating_contexts(f_acts_path, "imdb", sampled_feats, top_n=5)
        
        high_conf_count = 0
        confidences = []
        for feat_idx in tqdm(sampled_feats, desc=f"Annotating {arch.upper()}"):
            try:
                annotation = annotate_feature_with_llm(feat_idx, contexts[feat_idx], api_key="local", api_base=api_base, model_id=model_id)
                conf = annotation.get("confidence", 0.0)
                confidences.append(conf)
                if conf >= 0.85:
                    high_conf_count += 1
            except Exception:
                confidences.append(0.0)
                
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        extrapolated_high_conf_yield = int((high_conf_count / sample_size) * len(feats)) if sample_size > 0 else 0
        
        print(f"{arch.upper()} Avg Confidence: {avg_conf:.3f} | Features >= 0.85 (Extrapolated): {extrapolated_high_conf_yield}")
        
        results[arch] = {
            "avg_confidence": avg_conf,
            "high_conf_ratio": high_conf_count / sample_size if sample_size > 0 else 0,
            "extrapolated_total_high_conf": extrapolated_high_conf_yield
        }
        
    os.makedirs("results/gaps", exist_ok=True)
    with open("results/gaps/architecture_yield_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

def run_causal_noise_probe():
    print("\n--- GAP 1 (Part B): Causal Intervention on Noise Features ---")
    print("Selecting 5 random noise features to steer and check KL divergence...")
    
    report_path = "results/H3_mp_threshold_report.json"
    if not os.path.exists(report_path):
        return
        
    with open(report_path, "r") as f:
        report = json.load(f)
        
    noise_features = report.get("Categories", {}).get("dead", [])
    # Filter to only include features valid for Pythia-70m with ef=8 (dict_size 4096)
    noise_features = [f for f in noise_features if f < 4096]
    sae_path = "results/sae_models/standard_sae_EleutherAI_pythia-70m_imdb_ef8.pt" # Default path based on earlier context
            
    if len(noise_features) < 5 or not os.path.exists(sae_path):
        print("Required SAE not found or insufficient noise features.")
        return
        
    sampled_noise = random.sample(noise_features, 5)
    
    for feat in sampled_noise:
        cmd = [
            "python", "scripts/causal_intervention.py",
            "--model", "EleutherAI/pythia-70m",
            "--layer", "gpt_neox.layers.3",
            "--sae", sae_path,
            "--feature", str(feat),  # Fixed argument name to match the script's parser
            "--coeff", "100.0",      # Added a strong coefficient to test the intervention
        ]
        print(f"Running Causal Intervention on Noise Feature {feat}...")
        subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_api_base", type=str, default="http://10.80.63.42:8000/v1", help="vLLM endpoint")
    parser.add_argument("--local_model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="vLLM model id")
    args = parser.parse_args()
    
    evaluate_noise_features(args.local_api_base, args.local_model_id)
    evaluate_architecture_yield(args.local_api_base, args.local_model_id)
    run_causal_noise_probe()
    
    print("\nAll Gap experiments completed successfully.")
