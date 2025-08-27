#!/usr/bin/env python3
"""
Test the enhanced identity response system with clarification questions
"""

import json
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_identity_with_clarification():
    """Test that identity responses include clarification questions"""
    
    print("Testing Enhanced Identity System with Clarification Questions")
    print("=" * 70)
    
    analyzer = PersonalityAnalyzer()
    
    # Test case 1: Identity question in English with minimal personality data
    test_case_1 = {
        "id": 225985882206,
        "user_input": "Hello! I like working with data.",  # Minimal info
        "new_input": [
            {
                "question": "How do you interact with others?",
                "answer": "I work in teams sometimes."  # Minimal info
            },
            {
                "question": "How do you handle challenges?",
                "answer": "who are you"  # Identity question
            }
        ],
        "languages": "en"
    }
    
    print("Test Case 1: English Identity Question with Minimal Data")
    print(f"Input: Limited personality information provided")
    
    result = analyzer.analyze(
        id=test_case_1["id"],
        user_input=test_case_1["user_input"],
        new_input=test_case_1["new_input"],
        languages=test_case_1["languages"]
    )
    
    response_data = json.loads(result["content"])
    
    print(f"✅ Status: {response_data.get('status')}")
    print(f"✅ Identity Response: {response_data.get('description_identity', '')[:100]}...")
    print(f"✅ Missing Traits: {response_data.get('missing_traits', [])}")
    print(f"✅ Clarification Questions ({len(response_data.get('clarification_questions', []))}):")
    
    for i, question in enumerate(response_data.get('clarification_questions', []), 1):
        print(f"   {i}. {question}")
    
    # Verify behavior
    assert response_data["status"] == "identity", "Should have identity status"
    assert len(response_data["description_identity"]) > 0, "Should have identity response"
    assert len(response_data["missing_traits"]) > 0, "Should identify missing traits"
    assert len(response_data["clarification_questions"]) > 0, "Should provide clarification questions"
    
    print("✅ Test Case 1: PASSED")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test case 2: Arabic identity question
    test_case_2 = {
        "id": 225985882207,
        "user_input": "أنا أحب العمل مع البيانات",
        "new_input": [
            {
                "question": "كيف تتفاعل مع الآخرين؟",
                "answer": "أعمل في فرق أحياناً"
            },
            {
                "question": "كيف تتعامل مع التحديات؟",
                "answer": "من أنت"  # Arabic identity question
            }
        ],
        "languages": "ar"
    }
    
    print("Test Case 2: Arabic Identity Question")
    
    result = analyzer.analyze(
        id=test_case_2["id"],
        user_input=test_case_2["user_input"],
        new_input=test_case_2["new_input"],
        languages=test_case_2["languages"]
    )
    
    response_data = json.loads(result["content"])
    
    print(f"✅ Status: {response_data.get('status')}")
    print(f"✅ Identity Response: {response_data.get('description_identity', '')[:100]}...")
    print(f"✅ Missing Traits: {response_data.get('missing_traits', [])}")
    print(f"✅ Arabic Clarification Questions ({len(response_data.get('clarification_questions', []))}):")
    
    for i, question in enumerate(response_data.get('clarification_questions', []), 1):
        print(f"   {i}. {question}")
    
    # Verify Arabic content
    identity_response = response_data.get('description_identity', '')
    has_arabic_identity = any('\u0600' <= char <= '\u06FF' for char in identity_response)
    
    questions = response_data.get('clarification_questions', [])
    has_arabic_questions = any(
        any('\u0600' <= char <= '\u06FF' for char in q) for q in questions
    ) if questions else False
    
    assert response_data["status"] == "identity", "Should have identity status"
    assert has_arabic_identity, "Identity response should be in Arabic"
    assert has_arabic_questions, "Clarification questions should be in Arabic"
    
    print("✅ Test Case 2: PASSED")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test case 3: Identity question with more complete personality data
    test_case_3 = {
        "id": 225985882208,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights. I'm usually calm and logical in my approach.",
        "new_input": [
            {
                "question": "How do you interact with others?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you handle emotions?",
                "answer": "I try to stay emotionally balanced and use analytical thinking to work through challenges. I rarely get overwhelmed and prefer to approach problems systematically."
            },
            {
                "question": "What are your daily habits?",
                "answer": "what is begining"  # Identity question after more complete data
            }
        ],
        "languages": "en"
    }
    
    print("Test Case 3: Identity Question with More Complete Data")
    
    result = analyzer.analyze(
        id=test_case_3["id"],
        user_input=test_case_3["user_input"],
        new_input=test_case_3["new_input"],
        languages=test_case_3["languages"]
    )
    
    response_data = json.loads(result["content"])
    
    print(f"✅ Status: {response_data.get('status')}")
    print(f"✅ Identity Response: {response_data.get('description_identity', '')[:100]}...")
    print(f"✅ Missing Traits: {response_data.get('missing_traits', [])}")
    print(f"✅ Clarification Questions ({len(response_data.get('clarification_questions', []))}):")
    
    for i, question in enumerate(response_data.get('clarification_questions', []), 1):
        print(f"   {i}. {question}")
    
    # With more complete data, should have fewer missing traits
    missing_count = len(response_data.get('missing_traits', []))
    print(f"✅ Missing Traits Count: {missing_count} (should be fewer than previous tests)")
    
    assert response_data["status"] == "identity", "Should have identity status"
    assert missing_count < 4, "Should have fewer missing traits with more complete data"
    
    print("✅ Test Case 3: PASSED")
    
    print("\n🎉 All enhanced identity system tests passed!")

def test_clarification_generation():
    """Test the clarification question generation methods"""
    
    print("\n\nTesting Clarification Generation Methods")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # Test missing trait analysis
    print("1. Testing missing trait analysis...")
    
    minimal_input = "I like data"
    minimal_new_input = [{"question": "Test", "answer": "I work sometimes"}]
    
    missing_traits = analyzer.analyze_missing_traits(minimal_input, minimal_new_input)
    print(f"✅ Missing traits for minimal input: {missing_traits}")
    assert len(missing_traits) > 0, "Should identify missing traits"
    
    # Test clarification generation in English
    print("\n2. Testing English clarification generation...")
    english_questions = analyzer.generate_clarification_questions(missing_traits, "en", max_questions=3)
    print(f"✅ English questions generated: {len(english_questions)}")
    for i, q in enumerate(english_questions, 1):
        print(f"   {i}. {q}")
    assert len(english_questions) > 0, "Should generate English questions"
    
    # Test clarification generation in Arabic
    print("\n3. Testing Arabic clarification generation...")
    arabic_questions = analyzer.generate_clarification_questions(missing_traits, "ar", max_questions=3)
    print(f"✅ Arabic questions generated: {len(arabic_questions)}")
    for i, q in enumerate(arabic_questions, 1):
        print(f"   {i}. {q}")
    
    # Verify Arabic content
    has_arabic = any(
        any('\u0600' <= char <= '\u06FF' for char in q) for q in arabic_questions
    ) if arabic_questions else False
    
    assert len(arabic_questions) > 0, "Should generate Arabic questions"
    assert has_arabic, "Questions should contain Arabic text"
    
    print("✅ Clarification generation tests passed!")

if __name__ == "__main__":
    test_identity_with_clarification()
    test_clarification_generation()
