#!/usr/bin/env python3
"""
Test script to verify the enhanced greeting and introduction detection functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_greeting_scenarios():
    """Test various greeting and introduction scenarios"""
    
    analyzer = PersonalityAnalyzer()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Arabic greeting with name and job",
            "text": "مرحبا انا وليد مهندس بيوميجات",
            "expected_name": "وليد",
            "expected_job": "مهندس",
            "expected_language": "arabic"
        },
        {
            "name": "Arabic name only",
            "text": "انا احمد",
            "expected_name": "احمد", 
            "expected_job": "",
            "expected_language": "arabic"
        },
        {
            "name": "Arabic job only",
            "text": "انا مهندس",
            "expected_name": "",
            "expected_job": "مهندس",
            "expected_language": "arabic"
        },
        {
            "name": "English greeting with name and job",
            "text": "Hi I am John, I'm an engineer",
            "expected_name": "John",
            "expected_job": "engineer", 
            "expected_language": "english"
        },
        {
            "name": "Non-introduction question",
            "text": "من أنت",
            "expected_name": "",
            "expected_job": "",
            "expected_language": "arabic"
        },
        {
            "name": "Non-introduction English",
            "text": "What's the weather like?",
            "expected_name": "",
            "expected_job": "",
            "expected_language": "english"
        }
    ]
    
    print("Testing Personal Introduction Detection")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: '{test_case['text']}'")
        
        # Test introduction detection
        has_intro, name, job, greeting = analyzer.detect_personal_introduction(test_case['text'])
        
        print(f"Result: has_intro={has_intro}, name='{name}', job='{job}'")
        print(f"Greeting: '{greeting}'")
        
        # Verify results
        success = True
        if has_intro:
            if name != test_case['expected_name']:
                print(f"❌ Name mismatch: expected '{test_case['expected_name']}', got '{name}'")
                success = False
            if job != test_case['expected_job']:
                print(f"❌ Job mismatch: expected '{test_case['expected_job']}', got '{job}'")
                success = False
            if greeting and len(greeting.strip()) > 0:
                print(f"✅ Greeting generated successfully")
            else:
                print(f"⚠️  No greeting generated")
        else:
            if test_case['expected_name'] or test_case['expected_job']:
                print(f"❌ Should have detected introduction but didn't")
                success = False
            else:
                print(f"✅ Correctly identified as non-introduction")
        
        if success and has_intro and (test_case['expected_name'] or test_case['expected_job']):
            print(f"✅ Test passed")
        elif success and not has_intro and not test_case['expected_name'] and not test_case['expected_job']:
            print(f"✅ Test passed")
        else:
            print(f"❌ Test failed")
        
        print("-" * 30)

def test_full_analyze_flow():
    """Test the full analyze flow with introductions"""
    
    analyzer = PersonalityAnalyzer()
    
    print("\n\nTesting Full Analyze Flow")
    print("=" * 50)
    
    # Test case 1: Arabic introduction
    print("\nTest 1: Arabic Introduction")
    try:
        result = analyzer.analyze(
            id=1,
            user_input="مرحبا انا وليد مهندس بيوميجات",
            new_input=[],
            languages="ar"
        )
        
        import json
        if result and "content" in result:
            content = json.loads(result["content"])
            print(f"Status: {content.get('status', 'N/A')}")
            print(f"Personal Greeting: '{content.get('personal_greeting', '')}'")
            print(f"Missing Traits: {content.get('missing_traits', [])}")
            print(f"Questions: {content.get('clarification_questions', [])}")
            
            if content.get('personal_greeting'):
                print("✅ Personal greeting included")
            else:
                print("❌ Personal greeting missing")
        else:
            print("❌ No valid result returned")
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
    
    # Test case 2: Continuation without new introduction
    print("\nTest 2: Follow-up Response")
    try:
        result = analyzer.analyze(
            id=1,
            user_input="مرحبا انا وليد مهندس بيوميجات",
            new_input=[
                {"question": "كيف تشعر عادةً في المواقف الصعبة؟", "answer": "أشعر بالهدوء وأحب التحليل"}
            ],
            languages="ar"
        )
        
        import json
        if result and "content" in result:
            content = json.loads(result["content"])
            print(f"Status: {content.get('status', 'N/A')}")
            print(f"Personal Greeting: '{content.get('personal_greeting', '')}'")
            print(f"Description Arabic: '{content.get('description_arabic', '')[:100]}...'")
            
            # In follow-up, greeting should be empty but user info should be preserved
            if not content.get('personal_greeting'):
                print("✅ No duplicate greeting in follow-up")
            else:
                print("⚠️  Greeting present in follow-up (might be expected)")
        else:
            print("❌ No valid result returned")
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")

if __name__ == "__main__":
    test_greeting_scenarios()
    test_full_analyze_flow()
