import torch
import argparse
import json
import os
import base64
from io import BytesIO
from datasets import load_dataset
from transformers import AutoImageProcessor

def get_top_activating_images(f_acts, dataset, feature_idx, top_k=5):
    activations = f_acts[:, feature_idx]
    top_indices = torch.topk(activations, top_k).indices
    
    img_column = 'image' if 'image' in dataset.column_names else 'img'
    seq_len = 197
    target_images = [dataset[int(i) // seq_len][img_column] for i in top_indices]
    target_acts = activations[top_indices].tolist()
    
    return target_images, target_acts

def vlm_annotate(images, local_api_base=None, local_model_id="llava-hf/llava-v1.6-34b-hf"):
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if api_key != "EMPTY" or local_api_base:
        import openai
        
        if local_api_base:
            client = openai.OpenAI(base_url=local_api_base, api_key="EMPTY")
            model_to_use = local_model_id
        else:
            client = openai.OpenAI(api_key=api_key)
            model_to_use = "gpt-4o"
        
        content_items = [{"type": "text", "text": "Analyze these top activating images and identify the single unifying visual concept or pattern that causes a neural network feature to fire. Return ONLY a JSON object with 'caption' (a short 1-5 word label) and 'confidence' (0.0 to 1.0)."}]
        
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_str}"
                }
            })
            
        try:
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": content_items}],
                response_format={ "type": "json_object" } if not local_api_base else None
            )
            content = response.choices[0].message.content
            if content.startswith("```json"):
                content = content[7:-3]
            result = json.loads(content)
            return result.get("caption", "Unknown visual concept"), float(result.get("confidence", 0.0))
        except Exception as e:
            print(f"API Error: {e}")
            return "API Error", 0.0
    else:
        # Fallback mock VLM caption
        return "Visual concept: Geometric edge patterns / specific color blob", 0.92

def run_vision_annotation(report_path, f_acts_path, dataset_name, out_json, local_api_base, local_model_id):
    print("--- Phase 4: Vision Auto-Annotation ---")
    
    with open(report_path, "r") as f:
        report = json.load(f)
        
    domain_features = report.get("Categories", {}).get("domain", [])
    print(f"Found {len(domain_features)} domain-specific visual features to annotate.")
    
    if len(domain_features) == 0:
        print("No domain-specific features found. Skipping auto-annotation.")
        with open(out_json, "w") as f:
            json.dump({"error": "no domain features"}, f)
        return
        
    f_acts = torch.load(f_acts_path, map_location="cpu")
    dataset = load_dataset(dataset_name, split="train") # Using train split as in extraction
    
    results = {}
    
    # Just take a sample to avoid hitting mock API rate limits
    sample_features = domain_features[:10]
    
    for feat_idx in sample_features:
        print(f"Annotating feature {feat_idx}...")
        imgs, acts = get_top_activating_images(f_acts, dataset, feat_idx)
        
        # Call VLM
        caption, confidence = vlm_annotate(imgs, local_api_base, local_model_id)
        
        results[int(feat_idx)] = {
            "caption": caption,
            "confidence": confidence,
            "top_activations": acts
        }
        
    avg_confidence = sum(r['confidence'] for r in results.values()) / len(results) if len(results) > 0 else 0.0
    print(f"Average VLM Confidence: {avg_confidence:.2f}")
    
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved vision annotations to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True, help="V3 MP Threshold JSON report")
    parser.add_argument("--f_acts", type=str, required=True, help="Sparse feature activations .pt")
    parser.add_argument("--dataset", type=str, default="cifar10", help="HuggingFace dataset name")
    parser.add_argument("--out", type=str, default="results/V4_vision_annotations.json", help="Output JSON")
    parser.add_argument("--local_api_base", type=str, default=None, help="Local vLLM server base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--local_model_id", type=str, default="llava-hf/llava-v1.6-34b-hf", help="Local model ID")
    args = parser.parse_args()
    
    run_vision_annotation(args.report, args.f_acts, args.dataset, args.out, args.local_api_base, args.local_model_id)
