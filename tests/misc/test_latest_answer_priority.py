#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test that the system correctly prioritizes the LATEST answer in conversation history
and doesn't get confused by older identity questions.
"""

import sys
import os
import json
from dotenv import load_dotenv

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Load environment variables
load_dotenv()

from personality_analyzer import PersonalityAnalyzer

def test_latest_answer_priority():
    """
    Test that when there are multiple answers in conversation history,
    the system prioritizes the LATEST answer and doesn't detect identity
    from older answers.
    """
    print("="*80)
    print("TESTING: Latest Answer Priority Over Historical Identity Questions")
    print("="*80)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case: Older answer is identity question, newer answer is about analysis method
    test_input = {
        "id": 102,
        "user_input": "مرحبا! انا احمد كيف حالك؟",
        "new_input": [
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "من انت وما هدفك؟"  # This is an OLD identity question
            },
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟", 
                "answer": "كيف تحلل"  # This is the LATEST answer about analysis method
            }
        ],
        "languages": "ar"
    }
    
    print(f"Input data:")
    print(json.dumps(test_input, ensure_ascii=False, indent=2))
    print()
    
    # Analyze the input
    result = analyzer.analyze(**test_input)
    
    print(f"Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    
    # Verify expectations
    print("VERIFICATION:")
    print("-" * 40)
    
    # The latest answer is about "كيف تحلل" (how do you analyze) which should be treated as:
    # 1. NOT an identity question (since it's asking about analysis method, not identity)
    # 2. Should be processed for personality traits
    # 3. Should NOT trigger identity response based on the old "من انت وما هدفك؟"
    
    expected_no_identity = result.get("description_identity") is None
    print(f"✓ Should NOT have identity response (latest answer is not identity): {expected_no_identity}")
    
    if not expected_no_identity:
        print(f"  ❌ ISSUE: Got identity response: {result.get('description_identity')}")
        print(f"  ❌ This suggests the system is looking at old identity question instead of latest answer")
    else:
        print(f"  ✅ CORRECT: No identity response detected")
    
    # The system should detect this as a greeting with the name "احمد"
    has_greeting = bool(result.get("personal_greeting_and_off_topic"))
    print(f"✓ Should have greeting response for 'مرحبا انا احمد': {has_greeting}")
    
    if has_greeting:
        greeting_text = result.get("personal_greeting_and_off_topic", "")
        has_name = "احمد" in greeting_text
        print(f"  ✓ Greeting mentions name 'احمد': {has_name}")
        if has_name:
            print(f"  ✅ CORRECT: Greeting with name recognition")
        else:
            print(f"  ⚠️  WARNING: Greeting doesn't mention the name")
    
    # The latest answer "كيف تحلل" should be processed for traits
    missing_traits = result.get("missing_traits", [])
    print(f"✓ Should have missing traits (incomplete personality): {len(missing_traits) > 0}")
    
    # Should have clarification questions for missing traits
    clarification = result.get("clarification_questions", [])
    has_clarification = len(clarification) > 0
    print(f"✓ Should have clarification questions: {has_clarification}")
    
    print()
    print("="*80)
    print("SUMMARY:")
    print("="*80)
    
    if expected_no_identity and has_greeting and len(missing_traits) > 0 and has_clarification:
        print("✅ SUCCESS: System correctly prioritized latest answer over historical identity question")
        print("✅ Latest answer 'كيف تحلل' was processed for personality traits, not identity")
        print("✅ Old identity question 'من انت وما هدفك؟' was correctly ignored")
        return True
    else:
        print("❌ FAILURE: System did not correctly prioritize latest answer")
        if not expected_no_identity:
            print("  - Still detecting identity from old conversation history")
        if not has_greeting:
            print("  - Failed to recognize greeting pattern")
        if len(missing_traits) == 0:
            print("  - Did not detect incomplete personality")
        if not has_clarification:
            print("  - Did not generate clarification questions")
        return False

if __name__ == "__main__":
    success = test_latest_answer_priority()
    if not success:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed! System correctly handles latest answer priority.")
