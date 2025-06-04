#!/usr/bin/env python3
import sys
from enhanced_phi4_evaluator import EnhancedPhi4Evaluator

def main():
    print("🎯 Phase 2: Phi4 Baseline Evaluation")
    print("=" * 40)
    
    evaluator = EnhancedPhi4Evaluator()
    report = evaluator.run_evaluation(sample_size=5)
    
    if report:
        print("\n✅ Success! Baseline established.")
        return 0
    else:
        print("\n❌ Failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
