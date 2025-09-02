#!/usr/bin/env python3
"""
Test various scenarios for contextual clarification questions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_clarification_scenarios():
    """Test different scenarios for clarification question generation"""
    
    print("🧪 Testing Multiple Clarification Scenarios")
    print("=" * 70)
    
    analyzer = PersonalityAnalyzer()
    
    scenarios = [
        {
            "name": "Arabic: Greeting + Identity in History",
            "data": {
                "id": 201,
                "user_input": "مرحبا! انا سارة",
                "new_input": [
                    {
                        "question": "أخبرني عن نفسك",
                        "answer": "من انت وما هدفك؟"
                    }
                ],
                "languages": "ar"
            },
            "expect": "contextual_arabic"
        },
        {
            "name": "English: Greeting + Identity in History", 
            "data": {
                "id": 202,
                "user_input": "Hi! I'm Alex",
                "new_input": [
                    {
                        "question": "Tell me about yourself",
                        "answer": "Who are you and what is your purpose?"
                    }
                ],
                "languages": "en"
            },
            "expect": "contextual_english"
        },
        {
            "name": "Arabic: Greeting with NO Identity in History",
            "data": {
                "id": 203,
                "user_input": "مرحبا! كيف الحال؟",
                "new_input": [
                    {
                        "question": "كيف حالك؟",
                        "answer": "أنا شخص منطوي وأحب القراءة"
                    }
                ],
                "languages": "ar"
            },
            "expect": "generic_arabic"
        },
        {
            "name": "English: Greeting with NO Identity in History",
            "data": {
                "id": 204,
                "user_input": "Hello there!",
                "new_input": [
                    {
                        "question": "How are you?",
                        "answer": "I'm introverted and enjoy reading"
                    }
                ],
                "languages": "en"
            },
            "expect": "generic_english"
        },
        {
            "name": "Arabic: Greeting with Empty History",
            "data": {
                "id": 205,
                "user_input": "مرحبا!",
                "new_input": [],
                "languages": "ar"
            },
            "expect": "generic_arabic"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Test {i}: {scenario['name']}")
        print("-" * 50)
        
        data = scenario['data']
        expect = scenario['expect']
        
        # Run analysis
        result = analyzer.analyze(
            id=data["id"],
            user_input=data["user_input"],
            new_input=data["new_input"],
            languages=data["languages"]
        )
        
        # Extract clarification question
        clarification = result['clarification_questions'][0] if result['clarification_questions'] else ""
        
        # Check expectations
        is_arabic = any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in clarification)
        is_generic_ar = "تخبرني المزيد عن نفسك" in clarification
        is_generic_en = "tell me more about yourself" in clarification.lower()
        is_contextual = not (is_generic_ar or is_generic_en) and len(clarification) > 20
        
        print(f"Input: {data['user_input']}")
        print(f"Clarification: {clarification}")
        
        # Validate based on expectation
        if "contextual" in expect:
            success = is_contextual and not (is_generic_ar or is_generic_en)
            lang_ok = (is_arabic and "arabic" in expect) or (not is_arabic and "english" in expect)
            print(f"Expected: Contextual question in {'Arabic' if 'arabic' in expect else 'English'}")
            print(f"Result: {'✅ PASS' if success and lang_ok else '❌ FAIL'}")
        elif "generic" in expect:
            success = is_generic_ar or is_generic_en
            lang_ok = (is_arabic and "arabic" in expect) or (not is_arabic and "english" in expect)
            print(f"Expected: Generic question in {'Arabic' if 'arabic' in expect else 'English'}")
            print(f"Result: {'✅ PASS' if success and lang_ok else '❌ FAIL'}")
        
        # Show additional info
        has_identity = result['description_identity'] is not None
        print(f"Identity Response: {'✅ YES' if has_identity else '❌ NO'}")
        
    print(f"\n" + "=" * 70)
    print(f"🏁 Clarification Scenarios Testing Complete!")

if __name__ == "__main__":
    test_clarification_scenarios()
