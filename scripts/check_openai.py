#!/usr/bin/env python3
"""
Check which OpenAI models your account has access to.
Run this to find out which model to use in your .env file.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if OpenAI package is installed
try:
    import openai
except ImportError:
    print("❌ OpenAI package not installed. Run: pip install openai")
    sys.exit(1)

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in .env file")
    sys.exit(1)

print(f"✓ Found API Key: {api_key[:20]}...")
print("\n🔍 Checking available models...\n")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

try:
    # List all models
    models = client.models.list()
    
    # Filter for GPT models that support chat/function calling
    chat_models = []
    for model in models.data:
        model_id = model.id.lower()
        if any(x in model_id for x in ['gpt-3', 'gpt-4', 'chatgpt']):
            chat_models.append(model.id)
    
    if not chat_models:
        print("❌ No GPT models found in your account!")
        print("\n💡 Your account might be:")
        print("   1. Free tier with limited access")
        print("   2. New account without billing setup")
        print("   3. Restricted project permissions")
        print("\n📝 Next steps:")
        print("   1. Add billing: https://platform.openai.com/account/billing")
        print("   2. Check project settings: https://platform.openai.com/settings")
        sys.exit(1)
    
    print("✅ Available GPT Models for Chat/Function Calling:\n")
    print("="*70)
    
    # Sort and categorize models
    gpt4_models = [m for m in chat_models if 'gpt-4' in m.lower()]
    gpt3_models = [m for m in chat_models if 'gpt-3' in m.lower()]
    other_models = [m for m in chat_models if m not in gpt4_models and m not in gpt3_models]
    
    if gpt4_models:
        print("\n🚀 GPT-4 Models (BEST for AI search):")
        for model in sorted(gpt4_models):
            print(f"   ✓ {model}")
    
    if gpt3_models:
        print("\n⚡ GPT-3.5 Models (Good for AI search):")
        for model in sorted(gpt3_models):
            print(f"   ✓ {model}")
    
    if other_models:
        print("\n📝 Other Chat Models:")
        for model in sorted(other_models):
            print(f"   ✓ {model}")
    
    print("\n" + "="*70)
    print("\n💡 RECOMMENDED MODEL FOR YOUR .ENV:")
    print("="*70)
    
    # Recommend the best available model
    if gpt4_models:
        # Prefer specific GPT-4 versions in order
        recommended = None
        preferences = ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-4-0613']
        
        for pref in preferences:
            matching = [m for m in gpt4_models if pref in m.lower()]
            if matching:
                recommended = matching[0]
                break
        
        if not recommended:
            recommended = gpt4_models[0]
        
        print(f"\n✨ Use this in your .env file:")
        print(f"\n   OPENAI_MODEL={recommended}")
        print(f"\n   Why: Best accuracy for function calling")
        
    elif gpt3_models:
        # Prefer gpt-3.5-turbo variants
        recommended = None
        for model in gpt3_models:
            if 'gpt-3.5-turbo' in model.lower():
                recommended = model
                break
        
        if not recommended:
            recommended = gpt3_models[0]
        
        print(f"\n✨ Use this in your .env file:")
        print(f"\n   OPENAI_MODEL={recommended}")
        print(f"\n   Why: Most widely available, good for AI search")
    
    else:
        recommended = chat_models[0]
        print(f"\n✨ Use this in your .env file:")
        print(f"\n   OPENAI_MODEL={recommended}")
    
    print("\n" + "="*70)
    
    # Test the recommended model
    print(f"\n🧪 Testing {recommended} with a sample query...")
    
    try:
        response = client.chat.completions.create(
            model=recommended,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello' if you can hear me."}
            ],
            max_tokens=10
        )
        
        print(f"✅ SUCCESS! Model works correctly.")
        print(f"   Response: {response.choices[0].message.content}")
        
        # Check if it supports function calling
        try:
            test_functions = client.chat.completions.create(
                model=recommended,
                messages=[{"role": "user", "content": "Test"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "description": "Test",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }],
                max_tokens=10
            )
            print(f"✅ Function calling: SUPPORTED")
        except Exception as e:
            if "tool" in str(e).lower() or "function" in str(e).lower():
                print(f"⚠️  Function calling: NOT SUPPORTED")
                print(f"    This model may have limited AI search capabilities")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
    
    print("\n" + "="*70)
    print("\n✅ NEXT STEPS:")
    print("="*70)
    print("\n1. Update your .env file with the recommended model above")
    print("2. Restart your application: docker-compose restart")
    print("3. Test the AI search endpoint")
    print("\n")

except openai.AuthenticationError:
    print("❌ Authentication Error: Invalid API key")
    print("\n📝 Check your API key at: https://platform.openai.com/api-keys")
    
except openai.PermissionDeniedError as e:
    print(f"❌ Permission Denied: {e}")
    print("\n📝 Your account may need:")
    print("   1. Billing setup: https://platform.openai.com/account/billing")
    print("   2. Usage limits increased")
    print("   3. Project permissions updated")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print(f"\nFull error: {type(e).__name__}: {str(e)}")