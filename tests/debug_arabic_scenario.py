#!/usr/bin/env python3
"""
Debug the full Arabic scenario step by step
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_arabic_scenario():
    analyzer = PersonalityAnalyzer()

    print("🔍 DEBUGGING ARABIC SCENARIO STEP BY STEP")
    print("=" * 60)

    # Test data
    test_data = {
        'id': 225985882206,
        'user_input': 'مرحبًا! أنا شخص أستمتع حقًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة.',
        'new_input': [
            {
                'question': 'كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟',
                'answer': 'أحب العمل ضمن فرق وغالبًا ما أجد نفسي أتولى أدوار القيادة بشكل طبيعي.'
            },
            {
                'question': 'كيف تتعامل عادةً مع عواطفك في المواقف الصعبة؟',
                'answer': 'من مطورك'
            }
        ],
        'languages': 'ar'
    }

    print("📋 Test data:")
    print(f"new_input length: {len(test_data['new_input'])}")
    for i, qa in enumerate(test_data['new_input']):
        print(f"  {i+1}. Q: {qa['question'][:50]}...")
        print(f"     A: {qa['answer']}")

    # Step 1: Check what the system extracts as the last answer
    if test_data['new_input']:
        last_qa = test_data['new_input'][-1]
        last_answer = last_qa.get("answer", "").strip()
        print(f"\n🔍 Step 1 - Last answer extracted: '{last_answer}'")

        # Step 2: Test detection on this last answer
        is_detected, category, response_data = analyzer.detect_identity_question(last_answer)
        print(f"🔍 Step 2 - Detection on last answer: {is_detected}, {category}")

        if is_detected and response_data:
            identity_response = analyzer.get_identity_response(response_data, test_data['languages'])
            print(f"🔍 Step 3 - Expected identity response: {identity_response[:100]}...")
        else:
            print("🔍 Step 3 - No identity response expected")

    # Step 4: Run full analysis
    print(f"\n🔍 Step 4 - Running full analysis...")
    result = analyzer.analyze(**test_data)
    response = json.loads(result['content'])

    print(f"Final status: {response.get('status')}")
    print(f"Final identity response: {response.get('description_identity', 'None')}")
    print(f"Final missing traits: {response.get('missing_traits', [])}")

    # Expected vs Actual
    print(f"\n📊 EXPECTED vs ACTUAL:")
    print("Expected:")
    print("  - Status: 'identity'")
    print("  - Identity response: Arabic predefined developer response")
    print("  - Missing traits: list of traits")
    print("  - Clarification questions: Arabic questions")
    
    print("Actual:")
    print(f"  - Status: '{response.get('status')}'")
    print(f"  - Identity response: '{response.get('description_identity', 'None')}'")
    print(f"  - Missing traits: {response.get('missing_traits', [])}")
    print(f"  - Clarification questions: {len(response.get('clarification_questions', []))}")

if __name__ == "__main__":
    debug_arabic_scenario()
