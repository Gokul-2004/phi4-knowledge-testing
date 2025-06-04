#!/usr/bin/env python3
"""
Simple Phi4 Test - Check if model responds at all
"""

import subprocess
import time

def test_simple_question():
    print("🧪 Testing Phi4 with simple question...")
    
    # Very simple question
    question = "Hello, what is 2+2?"
    
    print(f"❓ Question: {question}")
    print("🔄 Querying Phi4...")
    
    start_time = time.time()
    
    try:
        cmd = ['ollama', 'run', 'phi4:latest', question]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        if result.returncode == 0:
            response = result.stdout.strip()
            print(f"✅ Response received in {response_time}s")
            print(f"📝 Response: {response}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Simple question also timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_simple_question()
