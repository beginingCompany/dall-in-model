#!/usr/bin/env python3
"""
Test multiple identity questions in conversation history
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_multiple_identity_history():
    """Test the exact user scenario with multiple identity questions in history"""
    
    print("🔍 Testing Multiple Identity Questions in History")
    print("=" * 70)
    
    analyzer = PersonalityAnalyzer()
    
    # Test data from user with multiple identity questions
    test_data = {
        "id": 102,
        "user_input": "مرحبا! انا احمد كيف حالك؟",
        "new_input": [
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "من انت وما هدفك؟"
            },
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "كيف تحلل"
            }
        ],
        "languages": "ar"
    }
    
    print(f"📝 INPUT:")
    print(f"   User Input: {test_data['user_input']}")
    print(f"   Conversation History:")
    for i, item in enumerate(test_data['new_input'], 1):
        print(f"      {i}. Q: {item['question']}")
        print(f"         A: {item['answer']}")
    print(f"   Language: {test_data['languages']}")
    print("-" * 70)
    
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
    print(f"   Clarification Questions: {result['clarification_questions']}")
    
    # Check if identity was detected
    has_greeting = bool(result['personal_greeting_and_off_topic'])
    has_identity = result['description_identity'] is not None
    has_name_recognition = "أحمد" in result['personal_greeting_and_off_topic'] if has_greeting else False
    is_contextual = bool(result['clarification_questions']) and "تخبرني المزيد عن نفسك" not in result['clarification_questions'][0]
    
    print(f"\n✅ VERIFICATION:")
    print(f"   Greeting Detected: {'✅ YES' if has_greeting else '❌ NO'}")
    print(f"   Name Recognition: {'✅ YES' if has_name_recognition else '❌ NO'}")
    print(f"   Identity from History: {'✅ YES' if has_identity else '❌ NO'}")
    print(f"   Contextual Question: {'✅ YES' if is_contextual else '❌ NO'}")
    
    print(f"\n📊 ANALYSIS OF IDENTITY QUESTIONS:")
    print(f"   1st Answer: 'من انت وما هدفك؟' → should detect: who_are_you + purpose")
    print(f"   2nd Answer: 'كيف تحلل' → should detect: how_analyze")
    print(f"   Expected: Identity response from ANY of these questions")
    
    if has_identity:
        print(f"\n🎉 SUCCESS: Identity detected from conversation history!")
        print(f"   Identity Content: {result['description_identity'][:100]}...")
        print(f"   Contextual Question: {result['clarification_questions'][0] if result['clarification_questions'] else 'None'}")
    else:
        print(f"\n❌ ISSUE: No identity detected despite multiple identity questions in history")
        print(f"   Both answers contain identity questions that should trigger detection")

if __name__ == "__main__":
    test_multiple_identity_history()
