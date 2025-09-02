#!/usr/bin/env python3
"""
Test with exact formatting including the comma from user's request.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from app.personality_analyzer import PersonalityAnalyzer

def test_with_comma():
    """Test with the exact user_input including comma."""
    analyzer = PersonalityAnalyzer()
    
    print("=" * 70)
    print("TEST WITH EXACT COMMA FORMAT")
    print("=" * 70)
    
    print("\n🔍 Testing with comma in user_input")
    print("-" * 50)
    print("user_input: 'من انت وما هدفك؟,' (with comma)")
    print("latest answer: 'ما لون السماء' (off-topic)")
    
    result = analyzer.analyze(
        id=102,
        user_input="من انت وما هدفك؟,",  # Note the comma here
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
    
    # Check if it matches user's unexpected result
    if result['description_identity'] and not result['personal_greeting_and_off_topic']:
        print(f"\n❌ REPRODUCES USER'S ISSUE: Identity response returned")
        print(f"   Identity response: '{result['description_identity']}'")
        print(f"   This should be off-topic instead!")
    elif result['personal_greeting_and_off_topic'] and not result['description_identity']:
        print(f"\n✅ WORKING CORRECTLY: Off-topic response returned")
    else:
        print(f"\n❓ UNEXPECTED: Both or neither responses")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_with_comma()
