#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_identity_in_conversation():
    """Test identity detection when it appears as an answer in conversation"""
    
    analyzer = PersonalityAnalyzer()
    
    print("=== Testing Identity Detection in Conversation Context ===")
    
    # Test the exact scenario from the user
    test_data = {
        "id": 225985882206,
        "user_input": "مرحبًا! أنا شخص يستمتع حقًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أجد متعة كبيرة في اكتشاف الأنماط واستخلاص الرؤى.",
        "new_input": [
            {
                "question": "كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟",
                "answer": "أحب العمل ضمن فرق وغالبًا ما أجد نفسي أتولى أدوارًا قيادية بشكل طبيعي. أستمتع بتوجيه الزملاء الجدد وتيسير النقاشات الجماعية."
            },
            {
                "question": "كيف تتعامل عادةً مع عواطفك في المواقف الصعبة؟",
                "answer": "من انت"
            }
        ],
        "languages": "ar"
    }
    
    print("Input user_input:", test_data["user_input"])
    print("Conversation context:")
    for i, qa in enumerate(test_data["new_input"]):
        print(f"  {i+1}. Q: {qa['question']}")
        print(f"     A: {qa['answer']}")
    
    print("\n--- Testing identity detection on 'من انت' ---")
    
    # Test 1: Direct identity detection on "من انت"
    identity_response = analyzer.get_identity_response("من انت", "ar", analyzer.client)
    print(f"Direct identity detection result: {bool(identity_response)}")
    if identity_response:
        print(f"Identity response: {identity_response}")
    
    # Test 2: Full analysis
    print("\n--- Full analysis result ---")
    result = analyzer.analyze(**test_data)
    
    print(f"Status: {result['status']}")
    print(f"Description identity: {result.get('description_identity', 'None')}")
    print(f"Missing traits: {result.get('missing_traits', [])}")
    print(f"Clarification questions: {result.get('clarification_questions', [])}")
    
    # Test 3: Check if it's being treated as mid-conversation
    print("\n--- Debugging conversation context ---")
    print(f"Number of previous Q&A pairs: {len(test_data['new_input'])}")
    
    # Check if the answer "من انت" is being detected as identity trigger
    is_identity_trigger = analyzer._is_identity_trigger("من انت")
    print(f"Is 'من انت' detected as identity trigger: {is_identity_trigger}")

if __name__ == "__main__":
    test_identity_in_conversation()
