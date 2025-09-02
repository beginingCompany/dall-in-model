#!/usr/bin/env python3
"""
Test the exact user scenario to debug why identity is returned instead of off-topic.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from app.personality_analyzer import PersonalityAnalyzer

def test_exact_user_scenario():
    """Test the exact scenario provided by the user."""
    analyzer = PersonalityAnalyzer()
    
    print("=" * 70)
    print("EXACT USER SCENARIO TEST")
    print("=" * 70)
    
    print("\n🔍 DEBUGGING: User's exact request")
    print("-" * 50)
    print("user_input: 'من انت وما هدفك؟' (identity question)")
    print("latest answer: 'ما لون السماء' (off-topic)")
    print("EXPECTED: Off-topic response (not identity)")
    print("ACTUAL: Let's see...")
    
    result = analyzer.analyze(
        id=102,
        user_input="من انت وما هدفك؟",
        new_input=[
            {"question": "هل يمكنك أن تخبرني المزيد عن نفسك؟", "answer": "من انت وما هدفك؟"},
            {"question": "هل يمكنك أن تخبرني المزيد عن نفسك؟", "answer": "ما لون السماء"}
        ],
        languages="ar"
    )
    
    print(f"\n📋 RESULT:")
    print(f"  Personal greeting/off-topic: '{result['personal_greeting_and_off_topic']}'")
    print(f"  Description identity: '{result['description_identity']}'")
    print(f"  Status: {result['status']}")
    
    # Analysis
    print(f"\n🧐 ANALYSIS:")
    if result['personal_greeting_and_off_topic'] and not result['description_identity']:
        print("  ✅ CORRECT: Off-topic response returned (as expected)")
    elif result['description_identity'] and not result['personal_greeting_and_off_topic']:
        print("  ❌ INCORRECT: Identity response returned (not expected)")
        print("  🐛 BUG: Latest answer off-topic detection failed")
    else:
        print("  ❓ UNEXPECTED: Both or neither responses returned")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_exact_user_scenario()
