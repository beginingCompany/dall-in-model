#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test to verify LATEST answer priority with a non-identity latest answer
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

def test_non_identity_latest_answer():
    """
    Test with latest answer being a personality description, not identity question
    """
    print("="*80)
    print("TESTING: Latest Answer is Personality Description (NOT identity)")
    print("="*80)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case: Older answer is identity, newer answer is personality description
    test_input = {
        "id": 103,
        "user_input": "مرحبا كيف حالك؟",
        "new_input": [
            {
                "question": "هل يمكنك أن تخبرني المزيد عن نفسك؟",
                "answer": "من انت وما هدفك؟"  # This is an OLD identity question
            },
            {
                "question": "صف شخصيتك",
                "answer": "أنا شخص هادئ ومتفهم أحب القراءة"  # This is personality description (LATEST)
            }
        ],
        "languages": "ar"
    }
    
    print("Input data:")
    print(json.dumps(test_input, indent=2, ensure_ascii=False))
    print()
    
    result = analyzer.analyze(
        user_input=test_input["user_input"],
        id=test_input["id"],
        new_input=test_input["new_input"],
        languages=test_input["languages"]
    )
    
    print("Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    
    print("VERIFICATION:")
    print("----------------------------------------")
    
    # Should NOT have identity response since latest answer is personality description
    has_identity = bool(result.get('description_identity'))
    print(f"✓ Should NOT have identity response (latest is personality): {not has_identity}")
    if has_identity:
        print(f"  ❌ ISSUE: Got identity response: {result.get('description_identity')}")
        print(f"  ❌ This suggests system checked old identity question instead of latest personality answer")
    else:
        print(f"  ✅ CORRECT: No identity response generated")
    
    # Should have greeting response
    has_greeting = bool(result.get('personal_greeting_and_off_topic'))
    print(f"✓ Should have greeting response: {has_greeting}")
    
    # Should have personality analysis from the latest answer
    has_personality = bool(result.get('description_arabic') or result.get('description_english'))
    print(f"✓ Should have personality description: {has_personality}")
    if has_personality:
        print(f"  ✅ Personality description: {result.get('description_arabic', result.get('description_english', 'N/A'))}")
    
    print()
    
    if not has_identity and has_greeting:
        print("✅ SUCCESS: System correctly prioritized latest personality answer over old identity question")
    else:
        print("❌ FAILURE: System did not correctly handle latest answer priority")

if __name__ == "__main__":
    test_non_identity_latest_answer()
