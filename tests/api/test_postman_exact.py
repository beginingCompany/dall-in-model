#!/usr/bin/env python3
"""
Test the exact Postman scenario from comprehensive test collection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_postman_scenario():
    """Test the exact scenario from the Postman collection"""
    
    print("🧪 Testing EXACT Postman Scenario")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Exact test from Postman comprehensive collection
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
    
    print(f"📝 TEST INPUT:")
    print(f"   ID: {test_data['id']}")
    print(f"   User Input: {test_data['user_input']}")
    print(f"   Conversation: Q: {test_data['new_input'][0]['question']}")
    print(f"                 A: {test_data['new_input'][0]['answer']}")
    print(f"   Language: {test_data['languages']}")
    print("-" * 60)
    
    # Analyze
    result = analyzer.analyze(
        id=test_data["id"],
        user_input=test_data["user_input"],
        new_input=test_data["new_input"],
        languages=test_data["languages"]
    )
    
    print(f"🎯 ANALYSIS RESULT:")
    print(f"   ID: {result['id']}")
    print(f"   Status: {result['status']}")
    print(f"   Greeting Response: {result['personal_greeting_and_off_topic']}")
    print(f"   Identity Response: {result['description_identity']}")
    print(f"   English Description: {result['description_english']}")
    print(f"   Arabic Description: {result['description_arabic']}")
    print(f"   Missing Traits: {result['missing_traits']}")
    print(f"   Clarification Questions: {result['clarification_questions']}")
    print(f"   Tokens: {result['input_tokens']} input + {result['output_tokens']} output = {result['total_tokens']} total")
    
    # Validation
    print(f"\n✅ VALIDATION:")
    has_greeting = bool(result['personal_greeting_and_off_topic'])
    has_identity = result['description_identity'] is not None
    has_name_in_greeting = "أحمد" in result['personal_greeting_and_off_topic']
    
    print(f"   ✅ Greeting Detected: {'YES' if has_greeting else 'NO'}")
    print(f"   ✅ Name Recognition: {'YES' if has_name_in_greeting else 'NO'}")  
    print(f"   ✅ Identity from History: {'YES' if has_identity else 'NO'}")
    print(f"   ✅ Correct Status: {'YES' if result['status'] == 'incomplete' else 'NO'}")
    print(f"   ✅ Proper Format: {'YES' if result['description_identity'] != '' else 'NO'}")
    
    # Expected behavior check
    expected_behavior = all([
        has_greeting,
        has_name_in_greeting,
        has_identity,
        result['status'] == 'incomplete',
        result['description_identity'] is not None
    ])
    
    print(f"\n🎉 OVERALL RESULT: {'SUCCESS' if expected_behavior else 'NEEDS REVIEW'}")
    
    if expected_behavior:
        print("   All expected behaviors working correctly!")
        print("   - Greeting detected from current input ✅")
        print("   - Name 'أحمد' recognized in greeting ✅")
        print("   - Identity question detected from conversation history ✅")
        print("   - Both responses properly formatted ✅")
    else:
        print("   Some behaviors need attention")

if __name__ == "__main__":
    test_postman_scenario()
