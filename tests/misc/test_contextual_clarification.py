#!/usr/bin/env python3
"""
Test contextual clarification questions improvement
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_contextual_clarification():
    """Test that clarification questions are contextual and appropriate"""
    
    print("🎯 Testing Contextual Clarification Questions")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test the exact user scenario
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
    
    print(f"📝 INPUT:")
    print(f"   User Input: {test_data['user_input']}")
    print(f"   Conversation History: Q: {test_data['new_input'][0]['question']}")
    print(f"                         A: {test_data['new_input'][0]['answer']}")
    print(f"   Language: {test_data['languages']}")
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
    print(f"   Identity: {result['description_identity']}")
    print(f"   Clarification Questions: {result['clarification_questions']}")
    
    # Check if clarification is contextual
    clarification = result['clarification_questions'][0] if result['clarification_questions'] else ""
    is_generic = "تخبرني المزيد عن نفسك" in clarification or "tell me more about yourself" in clarification.lower()
    is_contextual = not is_generic and len(clarification) > 20
    
    print(f"\n✅ VALIDATION:")
    print(f"   Has Greeting: {'✅ YES' if result['personal_greeting_and_off_topic'] else '❌ NO'}")
    print(f"   Has Identity: {'✅ YES' if result['description_identity'] else '❌ NO'}")
    print(f"   Has Clarification: {'✅ YES' if clarification else '❌ NO'}")
    print(f"   Is Generic Question: {'❌ YES' if is_generic else '✅ NO'}")
    print(f"   Is Contextual: {'✅ YES' if is_contextual else '❌ NO'}")
    print(f"   In Arabic: {'✅ YES' if any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in clarification) else '❌ NO'}")
    
    if is_contextual and not is_generic:
        print(f"\n🎉 SUCCESS: Contextual clarification question generated!")
        print(f"   Question: {clarification}")
    else:
        print(f"\n❌ NEEDS IMPROVEMENT: Question is too generic")
        print(f"   Question: {clarification}")
    
    # Test English version for comparison
    print(f"\n" + "=" * 60)
    print(f"🔄 Testing English Version")
    
    english_test = {
        "id": 103,
        "user_input": "Hello! I'm John, how are you?",
        "new_input": [
            {
                "question": "Could you tell me more about yourself?",
                "answer": "Who are you and what is your purpose?"
            }  
        ],
        "languages": "en"
    }
    
    result_en = analyzer.analyze(
        id=english_test["id"],
        user_input=english_test["user_input"],
        new_input=english_test["new_input"],
        languages=english_test["languages"]
    )
    
    clarification_en = result_en['clarification_questions'][0] if result_en['clarification_questions'] else ""
    is_generic_en = "tell me more about yourself" in clarification_en.lower()
    is_contextual_en = not is_generic_en and len(clarification_en) > 20
    
    print(f"   English Clarification: {clarification_en}")
    print(f"   Is Generic: {'❌ YES' if is_generic_en else '✅ NO'}")
    print(f"   Is Contextual: {'✅ YES' if is_contextual_en else '❌ NO'}")

if __name__ == "__main__":
    test_contextual_clarification()
