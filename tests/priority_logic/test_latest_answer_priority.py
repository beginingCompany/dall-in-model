#!/usr/bin/env python3
"""
Test the latest answer priority logic with the new scenario.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_latest_answer_priority():
    """Test the new latest answer priority logic"""
    
    analyzer = PersonalityAnalyzer()
    
    # Test case from user: identity user_input + mixed new_input (identity + off-topic)
    test_data = {
        "id": 102,
        "user_input": "من انت وما هدفك؟",  # Identity question
        "new_input": [
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "من انت وما هدفك؟"  # Identity answer
            },
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "ما لون السماء"  # Off-topic answer (LATEST)
            }
        ],
        "languages": "ar"
    }
    
    print("="*70)
    print("TEST: Identity user_input + Mixed new_input (latest answer is off-topic)")
    print("="*70)
    print("Scenario:")
    print(f"  user_input: {test_data['user_input']} (identity)")
    print(f"  answer 1: {test_data['new_input'][0]['answer']} (identity)")
    print(f"  answer 2: {test_data['new_input'][1]['answer']} (off-topic) ← LATEST")
    print("\nExpected: Should prioritize LATEST answer (off-topic) over user_input (identity)")
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
        
        # Check if the response correctly prioritizes off-topic (latest answer)
        if result.get("personal_greeting_and_off_topic") and not result.get("description_identity"):
            print("\n✅ SUCCESS: Correctly prioritized latest answer (off-topic) over user_input (identity)")
            
            # Check if response is varied/casual
            response_text = result.get("personal_greeting_and_off_topic", "")
            if len(response_text.split()) < 20:  # Should be casual and brief
                print("✅ SUCCESS: Response is appropriately casual and varied")
            else:
                print("❌ INFO: Response might be too long for casual off-topic")
                
        elif result.get("description_identity") and not result.get("personal_greeting_and_off_topic"):
            print("❌ ISSUE: Incorrectly prioritized identity over off-topic")
            print("  Expected: Off-topic response from latest answer")
            print("  Got: Identity response")
            
        else:
            print("❌ ISSUE: Unexpected response combination")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Additional test: Latest answer is identity
    print("\n" + "="*70)
    print("TEST 2: Off-topic user_input + Identity latest answer")
    print("="*70)
    
    test_data_2 = {
        "id": 103,
        "user_input": "ما لون السماء؟",  # Off-topic
        "new_input": [
            {
                "question": "أخبرني عن نفسك",
                "answer": "من انت وما هدفك؟"  # Identity (LATEST)
            }
        ],
        "languages": "ar"
    }
    
    print("Scenario:")
    print(f"  user_input: {test_data_2['user_input']} (off-topic)")
    print(f"  latest answer: {test_data_2['new_input'][0]['answer']} (identity) ← LATEST")
    print("\nExpected: Should prioritize LATEST answer (identity) over user_input (off-topic)")
    print("\n" + "-"*30 + "\n")
    
    try:
        result_2 = analyzer.analyze(
            id=test_data_2["id"],
            user_input=test_data_2["user_input"],
            new_input=test_data_2["new_input"],
            languages=test_data_2["languages"]
        )
        
        print("Analysis Result:")
        print(f"  Personal greeting/off-topic: '{result_2.get('personal_greeting_and_off_topic', '')}'")
        print(f"  Description identity: '{result_2.get('description_identity', '')}'")
        
        if result_2.get("description_identity") and not result_2.get("personal_greeting_and_off_topic"):
            print("\n✅ SUCCESS: Correctly prioritized latest answer (identity) over user_input (off-topic)")
        else:
            print("❌ ISSUE: Did not correctly prioritize latest answer")
            
    except Exception as e:
        print(f"❌ ERROR in Test 2: {e}")

if __name__ == "__main__":
    test_latest_answer_priority()
