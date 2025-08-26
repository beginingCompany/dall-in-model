#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_identity_both_languages():
    """Test identity detection in both Arabic and English conversation contexts"""
    
    analyzer = PersonalityAnalyzer()
    
    print("=== Testing Identity Detection in Both Languages ===")
    
    # Test 1: Arabic (the user's case)
    print("\n--- Test 1: Arabic Identity in Conversation ---")
    arabic_data = {
        "id": 1,
        "user_input": "مرحبًا! أنا شخص يستمتع بالعمل مع البيانات.",
        "new_input": [
            {
                "question": "كيف تتعامل عادةً مع عواطفك؟",
                "answer": "من انت"
            }
        ],
        "languages": "ar"
    }
    
    result1 = analyzer.analyze(**arabic_data)
    print(f"Status: {result1['status']}")
    print(f"Identity response: {result1.get('description_identity', 'None')}")
    print(f"Clarification: {result1.get('clarification_questions', [])}")
    
    # Test 2: English
    print("\n--- Test 2: English Identity in Conversation ---")
    english_data = {
        "id": 2,
        "user_input": "Hello! I'm someone who enjoys working with data.",
        "new_input": [
            {
                "question": "How do you handle emotions?",
                "answer": "who are you"
            }
        ],
        "languages": "en"
    }
    
    result2 = analyzer.analyze(**english_data)
    print(f"Status: {result2['status']}")
    print(f"Identity response: {result2.get('description_identity', 'None')}")
    print(f"Clarification: {result2.get('clarification_questions', [])}")
    
    # Test 3: No identity question (normal case)
    print("\n--- Test 3: Normal Conversation (No Identity) ---")
    normal_data = {
        "id": 3,
        "user_input": "I'm an outgoing person who loves meeting new people.",
        "new_input": [
            {
                "question": "How do you handle stress?",
                "answer": "I usually stay calm and work through problems step by step."
            }
        ],
        "languages": "en"
    }
    
    result3 = analyzer.analyze(**normal_data)
    print(f"Status: {result3['status']}")
    print(f"Identity response: {result3.get('description_identity', 'None')}")
    print(f"Clarification: {result3.get('clarification_questions', [])}")

if __name__ == "__main__":
    test_identity_both_languages()
