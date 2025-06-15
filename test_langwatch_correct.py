#!/usr/bin/env python3
"""
LangWatch Correct Integration - Official Method
Based on LangWatch documentation
"""
import os
import langwatch
from openai import OpenAI

# Setup LangWatch with your API key
langwatch.setup(
    api_key=os.getenv("LANGWATCH_API_KEY")
)

# Create OpenAI client
client = OpenAI()

@langwatch.trace()
def test_kubernetes_question():
    """Test function with LangWatch tracing"""
    print("🤖 Making API call with proper LangWatch tracing...")
    
    # Use autotrack to automatically capture OpenAI calls
    langwatch.get_current_trace().autotrack_openai_calls(client)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Kubernetes expert."},
            {"role": "user", "content": "What are the main benefits of using Kubernetes Services? Explain in 3 sentences."}
        ],
        max_tokens=150,
        temperature=0.2
    )
    
    print("✅ API call completed!")
    print(f"📝 Response: {response.choices[0].message.content}")
    print(f"💰 Tokens: {response.usage.total_tokens}")
    print(f"💵 Cost: ${response.usage.total_tokens * 0.00015:.6f}")
    
    return response

def main():
    print("🔍 Testing LangWatch with Official Integration Method")
    print("=" * 60)
    
    # Check environment variables
    api_key = os.getenv("LANGWATCH_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ LANGWATCH_API_KEY not set")
        return False
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    print(f"✅ LangWatch API Key: {api_key[:15]}...")
    print(f"✅ OpenAI API Key: {openai_key[:15]}...")
    
    try:
        # Call the traced function
        response = test_kubernetes_question()
        
        print("\n🎯 SUCCESS!")
        print("📊 This call should now appear in your LangWatch dashboard")
        print("🔄 Refresh your dashboard to see the trace data")
        print("🌐 Dashboard: https://app.langwatch.ai/gpt-evaluation-zFksCQ")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ LangWatch integration working correctly!")
        print("💡 Your dashboard should now show data")
    else:
        print("\n❌ Integration failed - check the error above")
