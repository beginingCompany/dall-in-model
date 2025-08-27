#!/usr/bin/env python3
"""
Test Arabic identity detection
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_arabic_identity():
    analyzer = PersonalityAnalyzer()

    # Test the Arabic phrase
    arabic_phrase = 'من مطورك'
    print(f'Testing Arabic phrase: "{arabic_phrase}"')

    # Test detection
    is_detected, category, response_data = analyzer.detect_identity_question(arabic_phrase)
    print(f'Detection result: {is_detected}, {category}')

    if is_detected:
        identity_response = analyzer.get_identity_response(response_data, 'ar')
        print(f'Arabic response: {identity_response[:100]}...')
    else:
        print('❌ Not detected as identity question')

    # Test the full scenario
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

    result = analyzer.analyze(**test_data)
    response = json.loads(result['content'])

    print(f'\nFull test result:')
    print(f'Status: {response.get("status")}')
    print(f'Identity response: {response.get("description_identity", "None")}')

if __name__ == "__main__":
    test_arabic_identity()
