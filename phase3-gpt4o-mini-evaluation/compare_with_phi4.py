#!/usr/bin/env python3
"""Compare GPT-4o Mini results with Phi4 baseline"""

import json
import sys
from pathlib import Path

def load_results():
    """Load both Phi4 and GPT-4o Mini results"""
    gpt4o_path = "results/gpt4o_mini_latest.json"
    phi4_path = "../phase2-enhanced-phi4-baseline/results/phi4_baseline_latest.json"
    
    results = {}
    
    # Load GPT-4o Mini results
    if Path(gpt4o_path).exists():
        with open(gpt4o_path) as f:
            results['gpt4o'] = json.load(f)
        print("✅ GPT-4o Mini results loaded")
    else:
        print("❌ GPT-4o Mini results not found - run evaluation first")
        return None
    
    # Load Phi4 results
    if Path(phi4_path).exists():
        with open(phi4_path) as f:
            results['phi4'] = json.load(f)
        print("✅ Phi4 baseline results loaded")
    else:
        print("❌ Phi4 baseline not found")
        return None
    
    return results

def compare_models(results):
    """Generate detailed comparison"""
    print("\n🔍 MODEL COMPARISON")
    print("=" * 50)
    
    gpt4o = results['gpt4o']
    phi4 = results['phi4']
    
    # Overall scores
    gpt4o_score = gpt4o['quality_metrics']['overall_score']
    phi4_score = phi4['quality_metrics']['overall_score']
    
    print(f"📊 Overall Scores:")
    print(f"   GPT-4o Mini: {gpt4o_score:.3f}")
    print(f"   Phi4:        {phi4_score:.3f}")
    print(f"   Difference:  {gpt4o_score - phi4_score:+.3f}")
    
    # Performance metrics
    print(f"\n⚡ Performance:")
    print(f"   GPT-4o Mini: {gpt4o['performance_metrics']['avg_response_time']}s avg")
    print(f"   Phi4:        {phi4['performance_metrics']['avg_response_time']}s avg")
    
    # Kubernetes 1.32 knowledge (GPT-4o only)
    if 'k8s_132_analysis' in gpt4o:
        k8s_confidence = gpt4o['k8s_132_analysis']['overall_confidence']
        print(f"\n🔍 Kubernetes 1.32 Knowledge:")
        print(f"   GPT-4o Mini: {k8s_confidence}")
        print(f"   Phi4:        Not tested (requires recent training data)")

def main():
    print("🎯 GPT-4o Mini vs Phi4 Comparison")
    results = load_results()
    if results:
        compare_models(results)
    return 0

if __name__ == "__main__":
    sys.exit(main())
