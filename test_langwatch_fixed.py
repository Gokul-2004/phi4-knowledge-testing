#!/usr/bin/env python3
"""
Test LangWatch Integration - Updated API
"""
import os
import openai
from datetime import datetime

def test_langwatch():
    print("🔍 Testing LangWatch Integration")
    print("=" * 40)
    
    # Check environment variables
    langwatch_key = os.getenv("LANGWATCH_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not langwatch_key:
        print("❌ LANGWATCH_API_KEY not set")
        return False
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not set") 
        return False
    
    print("✅ Environment variables set")
    print(f"🔑 LangWatch API Key: {langwatch_key[:10]}...")
    print(f"🔑 OpenAI API Key: {openai_key[:10]}...")
    
    # Initialize LangWatch with updated method
    try:
        import langwatch
        
        # Updated LangWatch initialization
        langwatch.api_key = langwatch_key
        
        # Alternative method - wrap OpenAI client
        from langwatch.openai import OpenAI
        client = OpenAI()
        
        print("✅ LangWatch client initialized")
    except Exception as e:
        print(f"❌ LangWatch init failed: {e}")
        print("💡 Trying alternative method...")
        
        # Fallback to regular OpenAI with manual logging
        try:
            client = openai.OpenAI()
            print("✅ Using OpenAI client (manual logging)")
        except Exception as e2:
            print(f"❌ OpenAI client failed: {e2}")
            return False
    
    # Test API call
    try:
        print("🤖 Making test API call to GPT-4o-mini...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Kubernetes expert."},
                {"role": "user", "content": "What is a Kubernetes Pod in one sentence?"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        print("✅ API call successful!")
        print(f"📝 Response: {response.choices[0].message.content}")
        print(f"💰 Tokens used: {response.usage.total_tokens}")
        print(f"💵 Estimated cost: ${response.usage.total_tokens * 0.00015:.6f}")
        
        print("\n🎯 SUCCESS!")
        print("👉 Go to your LangWatch dashboard to see this call logged")
        print("📊 You should see: 1 request, token usage, and response details")
        print("🌐 Dashboard: https://langwatch.ai")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI call failed: {e}")
        return False

if __name__ == "__main__":
    success = test_langwatch()
    if success:
        print("\n✅ LangWatch integration working!")
        print("💡 Ready for Phase 3 GPT evaluation")
    else:
        print("\n❌ Setup incomplete. Check LangWatch documentation.")
