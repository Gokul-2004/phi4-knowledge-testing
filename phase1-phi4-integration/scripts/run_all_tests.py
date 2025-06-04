#!/usr/bin/env python3
"""Phase 1: Complete Test Suite Runner"""

import subprocess
import sys
import time
from pathlib import Path

def run_test(test_script, test_name):
    print(f"\n🧪 Running {test_name}")
    print('='*50)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, test_script], 
                              capture_output=False, text=True, timeout=400)
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED ({total_time}s)")
            return True, total_time
        else:
            print(f"❌ {test_name} FAILED ({total_time}s)")
            return False, total_time
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False, 0

def main():
    print("🎯 Phase 1: Phi4 Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("tests/simple_phi4_test.py", "Simple Phi4 Test"),
        ("tests/phi4_k8s_extended.py", "Kubernetes Knowledge Test"),
    ]
    
    results = []
    total_start = time.time()
    
    for test_script, test_name in tests:
        if Path(test_script).exists():
            success, test_time = run_test(test_script, test_name)
            results.append((test_name, success, test_time))
        else:
            print(f"⚠️  Test file not found: {test_script}")
            results.append((test_name, False, 0))
    
    total_time = round(time.time() - total_start, 2)
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, test_time in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name:<25} | {test_time:>6.1f}s")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Total time: {total_time}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("👉 Phase 1 complete - ready for Phase 2")
        return 0
    else:
        print(f"\n⚠️  {total-passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
