#!/usr/bin/env python3
"""
Debug the issues with identity responses
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def debug_issues():
    """Debug the specific issues reported"""
    
    print("🔍 DEBUGGING IDENTITY RESPONSE ISSUES")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case 1: Check detection for first example
    print("1. Testing first case - 'who are you'")
    is_identity, key, data = analyzer.detect_identity_question("who are you")
    print(f"   Detected: {is_identity}, Key: {key}")
    
    # Test missing trait analysis for the first case
    user_input_1 = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights."
    new_input_1 = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "who are you"
        }
    ]
    
    missing_traits = analyzer.analyze_missing_traits(user_input_1, new_input_1[:-1])  # Exclude identity question
    print(f"   Missing traits for case 1: {missing_traits}")
    
    # Test clarification generation
    questions = analyzer.generate_clarification_questions(missing_traits, "en", 2)
    print(f"   Generated questions: {len(questions)}")
    for i, q in enumerate(questions, 1):
        print(f"      {i}. {q}")
    
    print("\n" + "-" * 40)
    
    # Test case 2: Check Arabic detection
    print("2. Testing Arabic case - 'من انت' vs 'من أنت'")
    
    # Test various Arabic variations
    arabic_tests = ["من انت", "من أنت", "من  انت", "من  أنت"]
    for arabic_text in arabic_tests:
        is_identity, key, data = analyzer.detect_identity_question(arabic_text)
        print(f"   '{arabic_text}' -> Detected: {is_identity}, Key: {key}")
    
    # Test Arabic missing traits
    user_input_2 = "من انت"
    new_input_2 = []
    
    missing_traits_ar = analyzer.analyze_missing_traits(user_input_2, new_input_2)
    print(f"   Missing traits for Arabic case: {missing_traits_ar}")
    
    questions_ar = analyzer.generate_clarification_questions(missing_traits_ar, "ar", 2)
    print(f"   Generated Arabic questions: {len(questions_ar)}")
    for i, q in enumerate(questions_ar, 1):
        print(f"      {i}. {q}")
    
    print("\n" + "-" * 40)
    
    # Test the full analyze method for both cases
    print("3. Testing full analyze method")
    
    # Case 1
    print("\n   Case 1: English with data")
    result1 = analyzer.analyze(
        id=225985882206,
        user_input=user_input_1,
        new_input=new_input_1,
        languages="en"
    )
    
    response1 = json.loads(result1["content"])
    print(f"   Status: {response1.get('status')}")
    print(f"   Missing traits: {response1.get('missing_traits')}")
    print(f"   Questions: {len(response1.get('clarification_questions', []))}")
    
    # Case 2  
    print("\n   Case 2: Arabic minimal data")
    result2 = analyzer.analyze(
        id=65387652876,
        user_input="من انت",
        new_input=[],
        languages="ar"
    )
    
    response2 = json.loads(result2["content"])
    print(f"   Status: {response2.get('status')}")
    print(f"   Identity response: {response2.get('description_identity', '')[:100]}...")
    print(f"   Missing traits: {response2.get('missing_traits')}")
    print(f"   Questions: {len(response2.get('clarification_questions', []))}")
    
    # Check if it's using our predefined Arabic response
    expected_arabic = "أنا ماينس زيرو، جزء من مشروع BEGINING"
    actual_arabic = response2.get('description_identity', '')
    uses_predefined = expected_arabic in actual_arabic
    print(f"   Uses predefined Arabic response: {uses_predefined}")

if __name__ == "__main__":
    debug_issues()
