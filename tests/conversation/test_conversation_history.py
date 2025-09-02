#!/usr/bin/env python3
"""
Test conversation history identity detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_conversation_history_identity():
    """Test the exact user scenario with conversation history containing identity question"""
    
    print("🔍 Testing Conversation History Identity Detection")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test data from user
    test_data = {
        "id": 102,
        "user_input": "مرحبا! انا احمد كيف حالك؟",
        "new_input": [
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "من انت وما هدفك؟"
            }  
        ],
        "languages": "ar"
    }
    
    print(f"📝 USER INPUT: {test_data['user_input']}")
    print(f"📋 CONVERSATION HISTORY:")
    for item in test_data['new_input']:
        print(f"   Q: {item['question']}")
        print(f"   A: {item['answer']}")
    print("-" * 60)
    
    # Analyze
    result = analyzer.analyze(
        id=test_data["id"],
        user_input=test_data["user_input"],
        new_input=test_data["new_input"],
        languages=test_data["languages"]
    )
    
    print(f"🎯 RESULT:")
    print(f"   Status: {result['status']}")
    print(f"   Greeting: {result['personal_greeting_and_off_topic']}")
    print(f"   Identity Response: {result['description_identity']}")
    print(f"   English Description: {result['description_english']}")
    print(f"   Arabic Description: {result['description_arabic']}")
    
    # Verify expectations
    has_greeting = bool(result['personal_greeting_and_off_topic'])
    has_identity = result['description_identity'] is not None
    
    print(f"\n✅ VERIFICATION:")
    print(f"   Greeting Detected: {'✅ YES' if has_greeting else '❌ NO'}")
    print(f"   Identity Detected: {'✅ YES' if has_identity else '❌ NO'}")
    
    if has_identity:
        print(f"   Identity Content: {result['description_identity'][:100]}...")
    
    print(f"\n📊 EXPECTATION:")
    print(f"   Should detect greeting from current input: ✅ YES")
    print(f"   Should detect identity from conversation history: ✅ YES")
    
    if has_greeting and has_identity:
        print(f"\n🎉 SUCCESS: Both greeting and identity correctly detected!")
    else:
        print(f"\n❌ ISSUE: Missing detection")
        if not has_greeting:
            print(f"     - No greeting detected from current input")
        if not has_identity:
            print(f"     - No identity detected from conversation history")

if __name__ == "__main__":
    test_conversation_history_identity()
