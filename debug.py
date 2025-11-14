# test_save_directly.py
"""
Test if sessions can be saved directly, bypassing the API endpoint
This will tell us if the issue is in the endpoint or in the service layer
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db.redis_client import get_redis
from app.infrastructure.cache import RedisCache
from app.conversation.context_manager import ContextManager


async def test_direct_save():
    print("=" * 70)
    print("TEST: Direct Session Save (Bypass API)")
    print("=" * 70)
    print()
    
    # Step 1: Initialize dependencies (same as in endpoint)
    print("1️⃣ Initializing dependencies...")
    redis_client = get_redis()
    cache = RedisCache(redis_client)
    manager = ContextManager(cache)
    print("   ✅ Dependencies initialized")
    print()
    
    # Step 2: Create session
    print("2️⃣ Creating session...")
    context = await manager.create_session(user_id="direct_test")
    print(f"   ✅ Session created: {context.session_id}")
    print(f"   State: {context.state}")
    print(f"   User ID: {context.user_id}")
    print()
    
    # Step 3: Check Redis immediately
    print("3️⃣ Checking Redis immediately...")
    keys = await redis_client.keys("conversation:context:*")
    print(f"   Found {len(keys)} sessions in Redis")
    
    if keys:
        print("   ✅ Sessions are being saved!")
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            print(f"      - {key}")
    else:
        print("   ❌ No sessions found!")
        print()
        print("   🔍 Debugging...")
        
        # Try to get the session
        retrieved = await manager.get_context(context.session_id)
        if retrieved:
            print("   ⚠️  Session exists in manager but not in Redis keys!")
        else:
            print("   ❌ Session doesn't exist in manager either!")
        
        # Check the exact key
        exact_key = f"conversation:context:{context.session_id}"
        exists = await redis_client.exists(exact_key)
        print(f"   Exact key exists: {exists}")
        
        if exists:
            # Get the data
            data = await redis_client.get(exact_key)
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            print(f"   Data length: {len(data)} chars")
            print("   ✅ Data is there, but keys() didn't find it!")
        else:
            print("   ❌ Key doesn't exist at all!")
            print()
            print("   💡 This means save_context() is failing silently!")
            print("      Check context_manager.py save_context() for errors")
    
    print()
    
    # Step 4: Verify retrieval
    print("4️⃣ Testing retrieval...")
    retrieved = await manager.get_context(context.session_id)
    
    if retrieved:
        print(f"   ✅ Session retrieved")
        print(f"   Session ID matches: {retrieved.session_id == context.session_id}")
    else:
        print(f"   ❌ Could not retrieve session!")
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    
    # Final check
    final_keys = await redis_client.keys("conversation:context:*")
    
    if len(final_keys) > 0:
        print("✅ Sessions CAN be saved directly")
        print("   → Issue is likely in your API endpoint's get_conversation_service()")
        print("   → Check if redis_client = None in conversation.py line 74")
    else:
        print("❌ Sessions CANNOT be saved even directly")
        print("   → Issue is in ContextManager.save_context() or RedisCache")
        print("   → Run: python diagnose_save_issue.py for detailed analysis")
    
    print()


if __name__ == "__main__":
    asyncio.run(test_direct_save())