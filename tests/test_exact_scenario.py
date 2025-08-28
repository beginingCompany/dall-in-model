#!/usr/bin/env python3
"""
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
