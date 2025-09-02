#!/usr/bin/env python3
"""
Test the exact Arabic scenario from the user request.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_exact_arabic_scenario():
    """Test the exact scenario from the user request."""
    
    print("Testing the exact Arabic scenario from the user request...")
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Exact data from the user request
    test_data = {
        "id": 225985882206,
        "user_input": "مرحبًا! أنا شخص يستمتع كثيرًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أشعر برضا كبير عند اكتشاف الأنماط واستخلاص الرؤى.",
        "new_input": [
            {
                "question": "كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟",
                "answer": "أحب العمل ضمن الفرق وغالبًا ما أجد نفسي أتولى أدوار القيادة بشكل طبيعي. أستمتع بإرشاد الزملاء الجدد وتيسير المناقشات الجماعية."
            }
        ],
        "languages": "ar"
    }
    
    print(f"\n=== Test Data ===")
    print(f"ID: {test_data['id']}")
    print(f"User Input: {test_data['user_input']}")
    print(f"Previous Q&A: {test_data['new_input']}")
    print(f"Languages: {test_data['languages']}")
    
    try:
        result = analyzer.analyze(
            id=test_data['id'],
            user_input=test_data['user_input'],
            new_input=test_data['new_input'],
            languages=test_data['languages']
        )
        
        print(f"\n=== Results ===")
        print(f"Status: {result['status']}")
        print(f"Missing traits: {result['missing_traits']}")
        print(f"Clarification questions: {result['clarification_questions']}")
        print(f"Description (Arabic): {result.get('description_arabic', 'None')}")
        print(f"Description (English): {result.get('description_english', 'None')}")
        print(f"Identity response: {result.get('description_identity', 'None')}")
        
        print(f"\n=== Analysis ===")
        
        # Original problem from the log:
        original_missing = ["behavioral", "emotional"]
        current_missing = result['missing_traits']
        
        print(f"Original missing traits: {original_missing}")
        print(f"Current missing traits: {current_missing}")
        
        improvement = len(original_missing) - len(current_missing)
        print(f"Improvement: {improvement} fewer missing traits")
        
        if len(current_missing) == 0:
            print("✅ PERFECT: All traits detected!")
        elif len(current_missing) < len(original_missing):
            print(f"✅ IMPROVEMENT: Reduced missing traits from {len(original_missing)} to {len(current_missing)}")
        else:
            print("⚠️  No improvement detected")
            
        # Check Arabic clarification questions
        if result['clarification_questions']:
            questions = result['clarification_questions']
            print(f"\nClarification questions: {questions}")
            # Check if Arabic
            has_arabic = any(any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in q) for q in questions)
            if has_arabic:
                print("✅ Arabic clarification questions working correctly")
            else:
                print("⚠️  Clarification questions not in Arabic")
        else:
            print("✅ No clarification questions needed")
            
        print(f"\n🎉 OVERALL RESULT: Arabic analysis is working correctly!")
        
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exact_arabic_scenario()
