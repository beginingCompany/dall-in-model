#!/usr/bin/env python3
"""
<<<<<<< HEAD
Test the exact user scenario to identify why identity response is triggering when it should be complete personality analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_exact_user_scenario():
    """Test the exact scenario from user request"""
    
    print("=== Testing Exact User Scenario ===\n")
    analyzer = PersonalityAnalyzer()
    
    # Exact data from user request
    user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights."
    
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "who are you"
        },
        {
            "question": "How do you typically approach and handle complex problem-solving tasks?",
            "answer": "i analytical can solving the problems by analyze them"
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "I handle my emotions by practicing self-awareness and emotional regulation."
        }
    ]
    
    print("INPUT DATA:")
    print(f"User Input: {user_input}")
    print(f"Conversation with {len(new_input)} Q&A pairs:")
    for i, qa in enumerate(new_input, 1):
        print(f"  {i}. Q: {qa['question']}")
        print(f"     A: {qa['answer']}")
    
    print(f"\n{'='*60}")
    print("ANALYZING...")
    
    result = analyzer.analyze(
        id=225985882206,
        user_input=user_input,
        new_input=new_input,
        languages="en"
    )
    
    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    
    if result.get("description_identity"):
        print("❌ ISSUE: Identity response triggered when it shouldn't be")
        print(f"   Identity response: {result['description_identity']}")
        print("   SHOULD BE: Personality description in description_english")
    else:
        print("✅ No identity response (correct)")
    
    if result.get("description_english") and len(result["description_english"]) > 50:
        print("✅ Personality description provided")
    else:
        print("❌ ISSUE: No personality description provided")
        print("   SHOULD BE: Complete personality analysis")
    
    print(f"Status: {result.get('status')}")
    print(f"Missing traits: {result.get('missing_traits')}")
    
    # Debug: Check what traits should be detected
    print(f"\nDEBUG - Manual trait detection:")
    full_text = user_input + " " + " ".join([qa.get('answer', '') for qa in new_input])
    print(f"Full text: {full_text[:200]}...")
    
    for trait, pattern in analyzer.TRAIT_PATTERNS.items():
        if re.search(pattern, full_text.lower()):
            print(f"  ✅ {trait}: detected")
        else:
            print(f"  ❌ {trait}: not detected")

if __name__ == "__main__":
    import re
    test_exact_user_scenario()
=======
Test the exact scenario from the user's API request
"""

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.personality_analyzer import PersonalityAnalyzer

def test_exact_scenario():
    """Test the exact scenario with conversation history"""
    
    print("🧪 TESTING EXACT USER SCENARIO")
    print("=" * 40)
    
    analyzer = PersonalityAnalyzer()
    
    # Exact data from user's request
    test_case = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions. "
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "who are you"
            },
            {
                "question": "How do you typically approach and handle your emotions in challenging situations?",
                "answer": "انا المهندس احمد"
            }
        ],
        "languages": "ar"
    }
    
    print("Test data:")
    print(f"  User input: {test_case['user_input'][:50]}...")
    print(f"  New input items: {len(test_case['new_input'])}")
    print(f"  Last answer: '{test_case['new_input'][-1]['answer']}'")
    print(f"  Languages: {test_case['languages']}")
    print("-" * 40)
    
    result = analyzer.analyze(**test_case)
    response = json.loads(result["content"])
    
    print(f"Status: {response.get('status')}")
    print(f"Personal Greeting: '{response.get('personal_greeting', '')}'")
    print(f"Arabic Description: {response.get('description_arabic', '')}")
    print(f"Missing Traits: {response.get('missing_traits', [])}")
    print(f"Questions: {len(response.get('clarification_questions', []))}")
    
    # Check if personal greeting exists and is not empty
    greeting = response.get('personal_greeting', '')
    if greeting and len(greeting.strip()) > 0:
        print("\n✅ SUCCESS: Personal greeting generated!")
        print(f"   Greeting: {greeting}")
    else:
        print("\n❌ FAILED: Personal greeting is empty")
        
    print("\n" + "=" * 40)

if __name__ == "__main__":
    test_exact_scenario()
>>>>>>> f912c397f4608be37933b416c471652681384d61
