#!/usr/bin/env python3
"""
Direct test of the personality analyzer off-topic detection fix with varied responses.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_off_topic_direct():
    """Test the analyzer directly without the API"""
    
    analyzer = PersonalityAnalyzer()
    
    # Test case 1: The original scenario from user request
    test_data_1 = {
        "id": 102,
        "user_input": "مرحبا! انا احمد كيف حالك؟",
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
    print("TEST 1: Original scenario - Greeting + Off-topic answers")
    print("="*60)
    print("Input:")
    print(f"  user_input: {test_data_1['user_input']}")
    print(f"  new_input: {test_data_1['new_input']}")
    print("\n" + "-"*50 + "\n")
    
    try:
        result_1 = analyzer.analyze(
            id=test_data_1["id"],
            user_input=test_data_1["user_input"],
            new_input=test_data_1["new_input"],
            languages=test_data_1["languages"]
        )
        
        print("Analysis Result:")
        print(f"  Status: {result_1.get('status')}")
        print(f"  Personal greeting/off-topic: '{result_1.get('personal_greeting_and_off_topic', '')}'")
        print(f"  Description identity: '{result_1.get('description_identity', '')}'")
        print(f"  Missing traits: {result_1.get('missing_traits', [])}")
        print(f"  Clarification questions: {result_1.get('clarification_questions', [])}")
        
        # Check if the response is correct
        if result_1.get("personal_greeting_and_off_topic"):
            print("\n✅ SUCCESS: Analyzer detected off-topic content!")
            
            # Check if response is varied/casual (not the long formal response)
            response_text = result_1.get("personal_greeting_and_off_topic", "")
            if len(response_text.split()) < 20:  # Casual responses should be shorter
                print("✅ SUCCESS: Response appears to be casual and varied")
            else:
                print("❌ INFO: Response might be too formal (check if it's varied)")
                
            # Verify that description_identity is None or empty
            if not result_1.get("description_identity"):
                print("✅ SUCCESS: description_identity is correctly None/empty")
            else:
                print("❌ ISSUE: description_identity should be None for off-topic")
                
        else:
            print("\n❌ ISSUE: Analyzer did not properly detect off-topic content")
            
    except Exception as e:
        print(f"❌ ERROR in Test 1: {e}")
        import traceback
        traceback.print_exc()

    # Test case 2: Pure off-topic content (no greeting)
    test_data_2 = {
        "id": 103,
        "user_input": "What's the weather like today?",
        "new_input": [
            {
                "question": "Tell me about yourself",
                "answer": "What's your favorite color?"
            },
            {
                "question": "How do you feel?", 
                "answer": "I like pizza"
            }
        ],
        "languages": "en"
    }
    
    print("\n" + "="*60)
    print("TEST 2: Pure off-topic - No greeting")
    print("="*60)
    print("Input:")
    print(f"  user_input: {test_data_2['user_input']}")
    print(f"  new_input: {test_data_2['new_input']}")
    print("\n" + "-"*50 + "\n")
    
    try:
        result_2 = analyzer.analyze(
            id=test_data_2["id"],
            user_input=test_data_2["user_input"],
            new_input=test_data_2["new_input"],
            languages=test_data_2["languages"]
        )
        
        print("Analysis Result:")
        print(f"  Status: {result_2.get('status')}")
        print(f"  Personal greeting/off-topic: '{result_2.get('personal_greeting_and_off_topic', '')}'")
        print(f"  Description identity: '{result_2.get('description_identity', '')}'")
        
        # Check if casual response is generated
        if result_2.get("personal_greeting_and_off_topic"):
            response_text = result_2.get("personal_greeting_and_off_topic", "")
            print(f"\n✅ SUCCESS: Off-topic response generated")
            print(f"  Response length: {len(response_text.split())} words")
            if len(response_text.split()) < 15:  # Should be short and casual
                print("✅ SUCCESS: Response is appropriately casual and brief")
            
    except Exception as e:
        print(f"❌ ERROR in Test 2: {e}")

    # Test case 3: Multiple runs to check variation
    print("\n" + "="*60)
    print("TEST 3: Multiple runs to check response variation")
    print("="*60)
    
    test_data_3 = {
        "id": 104,
        "user_input": "Nice weather today!",
        "new_input": [
            {
                "question": "Tell me about yourself",
                "answer": "The sky is blue"
            }
        ],
        "languages": "en"
    }
    
    responses = []
    for i in range(3):
        try:
            result = analyzer.analyze(
                id=test_data_3["id"] + i,
                user_input=test_data_3["user_input"],
                new_input=test_data_3["new_input"],
                languages=test_data_3["languages"]
            )
            response = result.get("personal_greeting_and_off_topic", "")
            responses.append(response)
            print(f"  Run {i+1}: '{response}'")
        except Exception as e:
            print(f"  Run {i+1}: Error - {e}")
    
    # Check if responses are different
    unique_responses = set(responses)
    if len(unique_responses) > 1:
        print(f"\n✅ SUCCESS: Responses show variation ({len(unique_responses)} unique out of {len(responses)})")
    else:
        print(f"\n❌ INFO: Responses might not be varied enough")

if __name__ == "__main__":
    test_off_topic_direct()
