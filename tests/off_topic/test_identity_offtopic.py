#!/usr/bin/env python3
"""
Test the identity + off-topic scenario where user_input is identity question.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_identity_plus_offtopic():
    """Test when user_input is identity question but conversation has off-topic content"""
    
    analyzer = PersonalityAnalyzer()
    
    # Test case: user_input is identity question + off-topic answers in new_input
    test_data = {
        "id": 102,
        "user_input": "من انت وما هدفك؟",
        "new_input": [
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "من انت وما هدفك؟"
            },
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "ما لون السماء"
            }
        ],
        "languages": "ar"
    }
    
    print("="*60)
    print("TEST: Identity question + Off-topic conversation")
    print("="*60)
    print("Input:")
    print(f"  user_input: {test_data['user_input']} (identity question)")
    print(f"  new_input answers: ")
    for i, qa in enumerate(test_data['new_input']):
        print(f"    {i+1}. '{qa['answer']}' ({'identity' if 'انت' in qa['answer'] else 'off-topic'})")
    print("\n" + "-"*50 + "\n")
    
    try:
        result = analyzer.analyze(
            id=test_data["id"],
            user_input=test_data["user_input"],
            new_input=test_data["new_input"],
            languages=test_data["languages"]
        )
        
        print("Analysis Result:")
        print(f"  Status: {result.get('status')}")
        print(f"  Personal greeting/off-topic: '{result.get('personal_greeting_and_off_topic', '')}'")
        print(f"  Description identity: '{result.get('description_identity', '')}'")
        print(f"  Missing traits: {result.get('missing_traits', [])}")
        print(f"  Clarification questions: {result.get('clarification_questions', [])}")
        
        # Expected behavior: Should prioritize off-topic over identity
        if result.get("personal_greeting_and_off_topic"):
            print("\n✅ SUCCESS: System prioritized off-topic response over identity!")
            print("✅ This is correct for mid-conversation with off-topic content")
            
            # Check response is casual and brief
            response_text = result.get("personal_greeting_and_off_topic", "")
            if len(response_text.split()) < 15:
                print("✅ SUCCESS: Response is appropriately casual and brief")
            
            # Verify no identity response when off-topic takes priority
            if not result.get("description_identity"):
                print("✅ SUCCESS: No identity response when off-topic takes priority")
            else:
                print("❌ INFO: Identity response present - check if this is expected")
                
        elif result.get("description_identity"):
            print("\n❌ ISSUE: System returned identity response instead of off-topic")
            print("❌ Expected: Off-topic should take priority in mid-conversation")
            print(f"   Identity response: '{result.get('description_identity', '')}'")
            
        else:
            print("\n❌ ISSUE: No clear response type detected")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Additional test case - comparison
    print("\n" + "="*60)
    print("COMPARISON: Pure identity question (no conversation)")
    print("="*60)
    
    test_data_2 = {
        "id": 103,
        "user_input": "من انت وما هدفك؟",
        "new_input": [],  # No conversation history
        "languages": "ar"
    }
    
    try:
        result_2 = analyzer.analyze(
            id=test_data_2["id"],
            user_input=test_data_2["user_input"],
            new_input=test_data_2["new_input"],
            languages=test_data_2["languages"]
        )
        
        print("Result without conversation:")
        print(f"  Personal greeting/off-topic: '{result_2.get('personal_greeting_and_off_topic', '')}'")
        print(f"  Description identity: Present = {bool(result_2.get('description_identity'))}")
        
        if result_2.get("description_identity"):
            print("✅ SUCCESS: Pure identity question correctly returns identity response")
        
    except Exception as e:
        print(f"❌ ERROR in comparison test: {e}")

if __name__ == "__main__":
    test_identity_plus_offtopic()
