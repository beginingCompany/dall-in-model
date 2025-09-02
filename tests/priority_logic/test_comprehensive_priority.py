#!/usr/bin/env python3
"""
Comprehensive test for conversation priority logic.
Tests all scenarios: off-topic, identity, and personality-related latest answers.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from app.personality_analyzer import PersonalityAnalyzer

def test_comprehensive_scenarios():
    """Test various scenarios to ensure proper priority handling."""
    analyzer = PersonalityAnalyzer()
    
    print("=" * 70)
    print("COMPREHENSIVE PRIORITY TESTS")
    print("=" * 70)
    
    # Test 1: Latest answer is off-topic
    print("\n🧪 TEST 1: Latest answer is OFF-TOPIC")
    print("-" * 50)
    result1 = analyzer.analyze(
        id=201,
        user_input="أريد تحليل شخصيتي",  # personality request
        new_input=[
            {"question": "أخبرني عن نفسك", "answer": "أنا أحب الرياضة"},
            {"question": "ما رأيك؟", "answer": "ما لون السماء؟"}  # off-topic latest
        ],
        languages="ar"
    )
    
    print(f"Result: {result1['personal_greeting_and_off_topic']}")
    print(f"Identity: {result1['description_identity']}")
    
    if result1['personal_greeting_and_off_topic'] and not result1['description_identity']:
        print("✅ SUCCESS: Off-topic latest answer handled correctly")
    else:
        print("❌ FAIL: Off-topic latest answer not handled correctly")
    
    # Test 2: Latest answer is identity
    print("\n🧪 TEST 2: Latest answer is IDENTITY")
    print("-" * 50)
    result2 = analyzer.analyze(
        id=202,
        user_input="ما لون السماء؟",  # off-topic
        new_input=[
            {"question": "سؤال عادي", "answer": "نعم"},
            {"question": "أخبرني", "answer": "من أنت ومن صنعك؟"}  # identity latest
        ],
        languages="ar"
    )
    
    print(f"Result: {result2['personal_greeting_and_off_topic']}")
    print(f"Identity: {result2['description_identity']}")
    
    if result2['description_identity'] and not result2['personal_greeting_and_off_topic']:
        print("✅ SUCCESS: Identity latest answer handled correctly")
    else:
        print("❌ FAIL: Identity latest answer not handled correctly")
    
    # Test 3: Latest answer is personality-related (should continue analysis)
    print("\n🧪 TEST 3: Latest answer is PERSONALITY-RELATED")
    print("-" * 50)
    result3 = analyzer.analyze(
        id=203,
        user_input="أريد تحليل شخصيتي",  # personality request
        new_input=[
            {"question": "كيف تتفاعل مع الآخرين؟", "answer": "أنا اجتماعي جداً وأحب اللقاءات"}  # personality latest
        ],
        languages="ar"
    )
    
    print(f"Result: {result3['personal_greeting_and_off_topic']}")
    print(f"Identity: {result3['description_identity']}")
    print(f"Status: {result3['status']}")
    
    if not result3['personal_greeting_and_off_topic'] and not result3['description_identity'] and result3['status'] == 'incomplete':
        print("✅ SUCCESS: Personality latest answer continues analysis correctly")
    else:
        print("❌ FAIL: Personality latest answer not handled correctly")
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TESTS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_comprehensive_scenarios()
