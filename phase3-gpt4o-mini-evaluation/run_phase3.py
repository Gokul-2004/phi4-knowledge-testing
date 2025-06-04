#!/usr/bin/env python3
"""
Phase 3 Complete Runner - Orchestrates GPT-4o Mini evaluation
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_prerequisites():
    """Check if environment is ready"""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if we're in the right directory
    if not Path("gpt4o_mini_evaluator.py").exists():
        print("❌ Run this from phase3-gpt4o-mini-evaluation directory")
        print("💡 Expected files: gpt4o_mini_evaluator.py")
        return False
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set in environment")
        print("💡 You'll be prompted to enter it")
    else:
        print("✅ OpenAI API key found")
    
    # Check OpenAI package
    try:
        import openai
        print("✅ OpenAI package available")
    except ImportError:
        print("❌ OpenAI package not found")
        print("💡 Installing OpenAI package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
            print("✅ OpenAI package installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install OpenAI package")
            return False
    
    return True

def run_evaluation():
    """Run the GPT-4o Mini evaluation"""
    print("\n🎯 Running GPT-4o Mini Evaluation")
    print("=" * 50)
    
    try:
        # Run the main evaluator
        result = subprocess.run([
            sys.executable, 
            "gpt4o_mini_evaluator.py"
        ], timeout=600)  # 10 minute timeout
        
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False

def run_comparison():
    """Run comparison with Phi4 if possible"""
    print("\n🔍 Running Model Comparison")
    print("=" * 40)
    
    if not Path("results/gpt4o_mini_latest.json").exists():
        print("❌ GPT-4o Mini results not found - evaluation may have failed")
        return False
    
    try:
        result = subprocess.run([
            sys.executable,
            "compare_with_phi4.py"
        ], timeout=30)
        
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        print("⏰ Comparison timed out")
        return False
    except Exception as e:
        print(f"❌ Error running comparison: {e}")
        return False

def generate_github_summary():
    """Generate summary for GitHub commit"""
    print("\n📦 Generating GitHub Summary")
    print("=" * 40)
    
    try:
        import json
        
        # Load results
        with open("results/gpt4o_mini_latest.json") as f:
            results = json.load(f)
        
        overall_score = results['quality_metrics']['overall_score']
        k8s_confidence = results['k8s_132_analysis']['overall_confidence']
        avg_time = results['performance_metrics']['avg_response_time']
        
        # Load Phi4 for comparison
        phi4_path = "../phase2-enhanced-phi4-baseline/results/phi4_baseline_latest.json"
        phi4_score = "N/A"
        if Path(phi4_path).exists():
            with open(phi4_path) as f:
                phi4_data = json.load(f)
                phi4_score = phi4_data['quality_metrics']['overall_score']
        
        print("🎯 PHASE 3 RESULTS SUMMARY")
        print("=" * 30)
        print(f"📊 GPT-4o Mini Score: {overall_score:.3f}/1.000")
        print(f"🔍 K8s 1.32 Knowledge: {k8s_confidence}")
        print(f"⚡ Avg Response Time: {avg_time}s")
        print(f"📈 vs Phi4 Baseline: {phi4_score}")
        
        samples_count = len(list(Path("samples").glob("*.json")))
        print(f"💾 Samples Saved: {samples_count}")
        
        print(f"\n🔧 Suggested Git Commands:")
        print(f"cd ~/data-prep")
        print(f"git add phase3-gpt4o-mini-evaluation/")
        print(f"git commit -m \"feat(phase3): Add GPT-4o Mini evaluation and comparison")
        print(f"")
        print(f"- Overall score: {overall_score:.3f}/1.000")
        print(f"- K8s 1.32 knowledge: {k8s_confidence}")
        print(f"- Performance: {avg_time}s avg response time") 
        print(f"- Comparison with Phi4 baseline ({phi4_score})")
        print(f"- {samples_count} samples saved with caching")
        print(f"- Ready for knowledge distillation comparison\"")
        print(f"git push origin main")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return False

def main():
    print("🚀 Phase 3: GPT-4o Mini Complete Evaluation")
    print("=" * 60)
    print("This will:")
    print("1. Check environment and dependencies")
    print("2. Run Kubernetes 1.32 knowledge detection")
    print("3. Run baseline evaluation (5 questions)")
    print("4. Compare with Phi4 baseline")
    print("5. Generate GitHub-ready summary")
    print("")
    
    # Step 1: Prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met")
        return 1
    
    print("✅ Prerequisites satisfied")
    
    # Step 2: Run evaluation
    if not run_evaluation():
        print("\n❌ Evaluation failed")
        return 1
    
    print("✅ Evaluation completed successfully")
    
    # Step 3: Run comparison
    if not run_comparison():
        print("\n⚠️  Comparison failed, but evaluation succeeded")
        # Don't fail completely if comparison fails
    else:
        print("✅ Comparison completed")
    
    # Step 4: Generate summary
    if not generate_github_summary():
        print("\n⚠️  Could not generate GitHub summary")
    else:
        print("✅ GitHub summary generated")
    
    print(f"\n🎉 PHASE 3 COMPLETE!")
    print(f"📁 Results in: phase3-gpt4o-mini-evaluation/results/")
    print(f"💾 Samples in: phase3-gpt4o-mini-evaluation/samples/")
    print(f"👉 Ready for Phase 4: Knowledge distillation planning")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
