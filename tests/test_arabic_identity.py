<<<<<<< HEAD
from app.personality_analyzer import PersonalityAnalyzer

# Test if the Arabic identity trigger is detected
test_answer = 'من أنت'
is_trigger = PersonalityAnalyzer._is_identity_trigger(test_answer)
print(f'Arabic answer "من أنت" is identity trigger: {is_trigger}')

# Test with some variations
test_cases = [
    'من أنت',
    'من أنت؟',
    'who are you',
    'what are you',
    'I am emotional',
    'I handle emotions well'
]

for case in test_cases:
    result = PersonalityAnalyzer._is_identity_trigger(case)
    print(f'"{case}" -> Identity trigger: {result}')
=======
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
>>>>>>> f912c397f4608be37933b416c471652681384d61
