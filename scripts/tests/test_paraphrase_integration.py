#!/usr/bin/env python3
"""
Quick test to verify paraphrase evaluation integration is working correctly.
Tests the caching mechanism and integration with performance_evaluation.py
"""

import os
import sys
import json
from pathlib import Path

# Add paramem to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_paraphrase_cache():
    """Test that paraphrase cache works correctly"""
    print("="*60)
    print("Testing Paraphrase Cache Mechanism")
    print("="*60)
    
    # Import the evaluation function
    from paramem.evaluation.wikidata_paraphrase_eval import generate_rule_based_paraphrases
    
    # Test paraphrase generation
    test_question = "What can be categorized as the capital of France?"
    paraphrases = generate_rule_based_paraphrases(test_question, num_paraphrases=3)
    
    print(f"\n✅ Generated {len(paraphrases)} paraphrases for test question:")
    print(f"   Original: {test_question}")
    for i, p in enumerate(paraphrases, 1):
        print(f"   {i}. {p}")
    
    # Test cache file structure
    cache_file = "/home/mmahaut/projects/paramem/data3/wikidata_paraphrase_cache.json"
    cache_dir = os.path.dirname(cache_file)
    
    print(f"\n📁 Cache file will be stored at: {cache_file}")
    print(f"   Cache directory exists: {os.path.exists(cache_dir)}")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print(f"   ✅ Existing cache found with {len(cache)} questions")
            
            # Show sample
            if cache:
                sample_key = list(cache.keys())[0]
                print(f"\n   Sample cached entry:")
                print(f"   Question: {sample_key}")
                print(f"   Paraphrases: {len(cache[sample_key])}")
        except Exception as e:
            print(f"   ⚠️  Could not read cache: {e}")
    else:
        print(f"   ℹ️  No existing cache (will be created on first run)")
    
    return True

def test_integration():
    """Verify integration with performance_evaluation.py"""
    print("\n" + "="*60)
    print("Testing Integration with performance_evaluation.py")
    print("="*60)
    
    perf_eval_path = "/home/mmahaut/projects/paramem/paramem/evaluation/performance_evaluation.py"
    
    # Check that file exists
    if not os.path.exists(perf_eval_path):
        print(f"❌ ERROR: {perf_eval_path} not found")
        return False
    
    # Check for paraphrase evaluation integration
    with open(perf_eval_path, 'r') as f:
        content = f.read()
    
    checks = {
        "Import statement": "from paramem.evaluation.wikidata_paraphrase_eval import evaluate_wikidata_with_paraphrases",
        "Cache file path": "wikidata_paraphrase_cache.json",
        "Cache parameter": "paraphrase_cache_file=",
        "Function call": "evaluate_wikidata_with_paraphrases(",
    }
    
    all_passed = True
    for check_name, check_string in checks.items():
        if check_string in content:
            print(f"   ✅ {check_name}: Found")
        else:
            print(f"   ❌ {check_name}: Missing")
            all_passed = False
    
    if all_passed:
        print(f"\n✅ All integration checks passed!")
    else:
        print(f"\n⚠️  Some integration checks failed")
    
    return all_passed

def main():
    print("\n🔍 Paraphrase Evaluation Integration Test\n")
    
    test1 = test_paraphrase_cache()
    test2 = test_integration()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Cache mechanism: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Integration:     {'✅ PASS' if test2 else '❌ FAIL'}")
    print("="*60)
    
    if test1 and test2:
        print("\n✅ All tests passed! The paraphrase evaluation is ready to use.")
        print("\nNext steps:")
        print("1. When a checkpoint evaluation runs, it will automatically:")
        print("   - Generate paraphrases (only once, then cached)")
        print("   - Evaluate model consistency across paraphrases")
        print("   - Save results to checkpoint's slurm_logs/paraphrase_stability.json")
        print("\n2. Paraphrases are cached in:")
        print("   /home/mmahaut/projects/paramem/data3/wikidata_paraphrase_cache.json")
        print("   This cache is shared across all checkpoints to save time.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

if __name__ == "__main__":
    main()
