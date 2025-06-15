#!/usr/bin/env python3
"""
Simple verification that LangWatch is logging to dashboard
"""

import os
import sys
sys.path.append(os.path.expanduser('~'))
from langwatch_global import setup_langwatch, traced_openai_call
import openai
from datetime import datetime

def main():
    print("🔍 LangWatch Dashboard Verification Test")
    print("=" * 45)
    
    # Setup LangWatch
    if not setup_langwatch():
        return 1
    
    # Setup OpenAI
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Test with a unique identifier so we can find it in dashboard
    test_id = datetime.now().strftime("%H%M%S")
    prompt = f"Test {test_id}: What is a Kubernetes Pod? Answer in exactly 10 words."
    
    print(f"🧪 Test ID: {test_id}")
    print(f"📝 Prompt: {prompt}")
    print("🚀 Sending to GPT-4o mini...")
    
    try:
        response = traced_openai_call(
            client,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.3
        )
        
        if response:
            print(f"✅ SUCCESS! Test {test_id} completed")
            print(f"📊 Response: {response.choices[0].message.content}")
            print(f"🎯 Tokens: {response.usage.total_tokens}")
            
            print(f"\n📈 VERIFICATION STEPS:")
            print(f"1. Go to: https://langwatch.ai/")
            print(f"2. Login to your dashboard")
            print(f"3. Look for a trace with:")
            print(f"   - Prompt containing 'Test {test_id}'")
            print(f"   - Model: gpt-4o-mini")
            print(f"   - Timestamp: around {datetime.now().strftime('%H:%M:%S')}")
            print(f"   - Token count: {response.usage.total_tokens}")
            
            print(f"\n🎉 If you see this trace in LangWatch dashboard,")
            print(f"   your integration is working perfectly!")
            
            return 0
        else:
            print("❌ No response received")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
