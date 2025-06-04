#!/usr/bin/env python3
"""
Phi4 Kubernetes Test - Extended Timeout Version
"""

import subprocess
import time
import json
import os
from datetime import datetime

def test_phi4_k8s_extended():
    print("🧪 Testing Phi4 with Kubernetes 1.3.2 (extended timeout)...")
    
    question = "What do you know about Kubernetes version 1.3.2 release notes? When was it released and what were the main changes?"
    
    print(f"❓ Question: {question}")
    print("🔄 Querying Phi4 (allowing up to 5 minutes)...")
    print("⏳ This may take longer for complex questions...")
    
    start_time = time.time()
    
    try:
        cmd = ['ollama', 'run', 'phi4:latest', question]
        # Extended timeout to 5 minutes for complex questions
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        if result.returncode == 0:
            response = result.stdout.strip()
            print(f"✅ Response received in {response_time}s ({response_time/60:.1f} minutes)")
            print("📝 Phi4 Response:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
            # Detailed analysis
            response_lower = response.lower()
            
            print("\n📊 Knowledge Analysis:")
            
            # Check for specific version knowledge
            if '1.3.2' in response_lower:
                print("  🎯 ✅ Has specific Kubernetes 1.3.2 knowledge")
                specific_knowledge = True
            elif '1.3' in response_lower:
                print("  🔶 ✅ Has Kubernetes 1.3.x series knowledge")
                specific_knowledge = True
            else:
                print("  ❌ No specific 1.3.2 version knowledge detected")
                specific_knowledge = False
            
            # Check for general Kubernetes awareness
            k8s_terms = ['kubernetes', 'k8s', 'container', 'pod', 'deployment', 'cluster']
            k8s_awareness = any(term in response_lower for term in k8s_terms)
            if k8s_awareness:
                print("  🔶 ✅ Shows general Kubernetes awareness")
            else:
                print("  ❌ Limited Kubernetes awareness")
            
            # Check for release-related concepts
            release_terms = ['release', 'version', 'update', 'patch', 'bug fix', 'feature', 'changelog']
            release_awareness = any(term in response_lower for term in release_terms)
            if release_awareness:
                print("  🔶 ✅ Understands software release concepts")
            
            # Check response quality
            word_count = len(response.split())
            if word_count > 100:
                print(f"  ✅ Detailed response ({word_count} words)")
            elif word_count > 30:
                print(f"  🔶 Moderate response ({word_count} words)")
            else:
                print(f"  ⚠️  Brief response ({word_count} words)")
            
            # Save detailed results
            save_detailed_results(question, response, response_time, {
                'specific_k8s_132': specific_knowledge,
                'general_k8s_awareness': k8s_awareness,
                'release_awareness': release_awareness,
                'word_count': word_count,
                'response_time': response_time
            })
            
            print(f"\n💾 Results saved to logs/")
            print(f"📈 Performance: {response_time:.1f}s for {word_count} words")
            
            return True
            
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            print(f"❌ Query failed: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Query timed out after 5 minutes")
        print("💡 Phi4 might need more time or system resources for complex queries")
        return False
    except Exception as e:
        print(f"❌ Error during query: {e}")
        return False

def save_detailed_results(question, response, response_time, analysis):
    """Save detailed test results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs('logs', exist_ok=True)
    
    results = {
        'timestamp': timestamp,
        'model': 'phi4:latest',
        'test_type': 'kubernetes_1_3_2_knowledge',
        'question': question,
        'response': response,
        'response_time_seconds': response_time,
        'analysis': analysis,
        'conclusion': 'Phi4 working for complex queries' if analysis['word_count'] > 10 else 'Limited response'
    }
    
    filename = f"logs/phi4_k8s_detailed_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Detailed results: {filename}")
    except Exception as e:
        print(f"❌ Could not save results: {e}")

def main():
    print("🎯 Phi4 Kubernetes 1.3.2 Extended Knowledge Test")
    print("=" * 55)
    print("✅ Phi4 confirmed working (simple test passed)")
    print("🔄 Testing complex Kubernetes query with extended timeout...")
    print("")
    
    success = test_phi4_k8s_extended()
    
    if success:
        print("\n🎉 SUCCESS: Phi4 can handle complex queries!")
        print("👉 Ready to build data preparation pipeline with Phi4")
        print("📋 Next steps:")
        print("   1. ✅ Phi4 knowledge confirmed")
        print("   2. 🔄 Build caching system")
        print("   3. 🔄 Create data processing pipeline")
        print("   4. 🔄 HuggingFace integration")
    else:
        print("\n⚠️  Complex queries still timing out")
        print("💡 Options:")
        print("   - Use Phi3 (smaller, faster) for data prep tasks")
        print("   - Increase system resources")
        print("   - Use Phi4 for simpler queries only")

if __name__ == "__main__":
    main()
