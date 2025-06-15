#!/usr/bin/env python3
"""
Quick test to verify LangWatch is logging GPT-4o mini calls
"""

import os
import openai
import langwatch
from datetime import datetime

# Initialize LangWatch
langwatch.configure(api_key=os.getenv('LANGWATCH_API_KEY'))

# Initialize OpenAI
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@langwatch.trace()
def quick_gpt_test():
    """Quick GPT-4o mini test with LangWatch tracking"""
    
    print("🧪 Quick LangWatch Test")
    print("=" * 30)
    
    # Simple test prompt
    prompt = "What is Kubernetes in one sentence?"
    
    print(f"📝 Prompt: {prompt}")
    print("🔄 Calling GPT-4o mini...")
    
    start = datetime.now()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.5
    )
    
    end = datetime.now()
    duration = (end - start).total_seconds()
    
    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens
    
    print(f"✅ Response: {answer}")
    print(f"⏱️  Time: {duration:.2f}s")
    print(f"🎯 Tokens: {tokens}")
    
    # Add metadata to LangWatch trace
    langwatch.get_current_trace().update(
        metadata={
            "test_type": "quick_verification",
            "response_time": duration,
            "tokens_used": tokens,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    return {"success": True, "tokens": tokens, "time": duration}

if __name__ == "__main__":
    try:
        result = quick_gpt_test()
        
        print(f"\n🎉 Test completed!")
        print(f"📊 Check your LangWatch dashboard at: https://langwatch.ai/")
        print(f"🔍 Look for the trace with metadata 'quick_verification'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure your API keys are set:")
        print(f"   - OPENAI_API_KEY")
        print(f"   - LANGWATCH_API_KEY")
