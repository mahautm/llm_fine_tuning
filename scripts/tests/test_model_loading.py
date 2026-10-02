#!/usr/bin/env python3
"""
Test script to validate model loading and memory requirements before full training.
Run this first to ensure your configuration will work.
"""

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

app = typer.Typer()


@app.command()
def main(
    model_name: str = typer.Option(..., help="HuggingFace model name to test"),
    use_lora: bool = typer.Option(True, help="Test LoRA configuration"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha"),
    use_8bit: bool = typer.Option(False, help="Test 8-bit quantization"),
):
    """
    Test model loading and estimate memory requirements.
    
    Examples:
        # Test 7B model with LoRA
        srun --ntasks=1 python scripts/tests/test_model_loading.py --model-name meta-llama/Llama-2-7b-hf
        
        # Test 13B model with LoRA
        srun --ntasks=1 python scripts/tests/test_model_loading.py --model-name meta-llama/Llama-2-13b-hf
        
        # Test with 8-bit quantization
        srun --ntasks=1 python scripts/tests/test_model_loading.py --model-name meta-llama/Llama-2-7b-hf --use-8bit
    """
    print("="*60)
    print(f"Testing Model: {model_name}")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires GPU.")
        return
    
    print(f"\n📊 Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    Total Memory: {mem_total:.2f} GB")
    
    print("\n1️⃣  Loading base model...")
    
    try:
        if use_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("✅ Loaded with 8-bit quantization")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print("✅ Loaded with bfloat16")
        
        # Calculate model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Statistics:")
        print(f"  Total parameters: {total_params/1e9:.2f}B")
        print(f"  Trainable parameters: {trainable_params/1e9:.2f}B")
        
        # Memory usage
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        
        print(f"\n💾 Memory Usage (Base Model):")
        print(f"  Allocated: {mem_allocated:.2f} GB")
        print(f"  Reserved: {mem_reserved:.2f} GB")
        
        if use_lora:
            print(f"\n2️⃣  Adding LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            model = get_peft_model(model, lora_config)
            print("✅ LoRA adapters added")
            
            model.print_trainable_parameters()
            
            # Memory usage after LoRA
            mem_allocated_lora = torch.cuda.memory_allocated() / 1e9
            mem_reserved_lora = torch.cuda.memory_reserved() / 1e9
            
            print(f"\n💾 Memory Usage (With LoRA):")
            print(f"  Allocated: {mem_allocated_lora:.2f} GB")
            print(f"  Reserved: {mem_reserved_lora:.2f} GB")
            print(f"  Additional (LoRA): {mem_allocated_lora - mem_allocated:.2f} GB")
        
        # Test forward pass
        print(f"\n3️⃣  Testing forward pass...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_text = "This is a test sentence to verify the model works."
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✅ Forward pass successful")
        
        mem_after_forward = torch.cuda.memory_allocated() / 1e9
        print(f"  Memory after forward: {mem_after_forward:.2f} GB")
        
        # Estimate training memory
        print(f"\n📈 Estimated Training Memory:")
        print(f"  Current (inference): {mem_after_forward:.2f} GB")
        
        # Training typically requires ~3-4x memory (gradients + optimizer states)
        if use_lora:
            multiplier = 2.5  # LoRA is more memory efficient
        else:
            multiplier = 4.0  # Full fine-tuning
        
        estimated_training = mem_after_forward * multiplier
        print(f"  Estimated (training): {estimated_training:.2f} GB")
        print(f"  Multiplier used: {multiplier}x")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        total_gpu_mem = sum(
            torch.cuda.get_device_properties(i).total_memory / 1e9 
            for i in range(torch.cuda.device_count())
        )
        
        if estimated_training > total_gpu_mem * 0.9:
            print("  ⚠️  Warning: Estimated memory exceeds available GPU memory!")
            print("  Suggestions:")
            if not use_lora:
                print("    - Use LoRA instead of full fine-tuning")
            if not use_8bit:
                print("    - Try 8-bit quantization (--use-8bit)")
            print("    - Reduce batch size to 1")
            print("    - Increase gradient accumulation")
            print("    - Use more GPUs with FSDP")
        else:
            print(f"  ✅ Configuration should work!")
            available_headroom = total_gpu_mem - estimated_training
            print(f"  Available headroom: {available_headroom:.2f} GB")
            
            if use_lora:
                suggested_batch_size = int(available_headroom / (mem_after_forward * 0.5))
                suggested_batch_size = max(1, min(suggested_batch_size, 8))
                print(f"  Suggested batch size per device: {suggested_batch_size}")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis might be due to:")
        print("  - Insufficient GPU memory")
        print("  - Model not found or access denied")
        print("  - Missing dependencies")
        return 1
    
    return 0


if __name__ == "__main__":
    app()
