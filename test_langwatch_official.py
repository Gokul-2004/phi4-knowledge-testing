#!/usr/bin/env python3
"""
LangWatch Proper Integration Test
Based on their official docs
"""
import os
import langwatch

# Set up LangWatch with your API key
langwatch.api_key = os.getenv("LANGWATCH_API_KEY")

# Import the LangWatch OpenAI wrapper
from langwatch.openai import OpenAI

def test_proper_integration():
    print("🔍 Testing LangWatch with Official Method")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("LANGWATCH_API_KEY")
    if not api_key:
        print("❌ LANGWATCH_API_KEY not set")
        return False
    
    print(f"✅ Using API key: {api_key[:15]}...")
    
    try:
        # Use LangWatch's OpenAI wrapper
        client = OpenAI()
        
        print("🤖 Making API call with LangWatch tracking...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful Kubernetes expert."},
                {"role": "user", "content": "Explain what a Kubernetes Service is in 2 sentences."}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        print("✅ API call completed!")
        print(f"📝 Response: {response.choices[0].message.content}")
        print(f"💰 Tokens: {response.usage.total_tokens}")
        print(f"💵 Cost: ${response.usage.total_tokens * 0.00015:.6f}")
        
        print("\n🎯 SUCCESS!")
        print("📊 This should now appear in your LangWatch dashboard")
        print("🔄 Refresh your dashboard page to see the data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_proper_integration()
    if success:
        print("\n✅ Integration working!")
        print("👉 Check dashboard: https://app.langwatch.ai/gpt-evaluation-zFksCQ")
    else:
        print("\n❌ Integration failed")
