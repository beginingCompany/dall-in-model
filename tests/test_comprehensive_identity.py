"""
Comprehensive test to verify all identity detection scenarios work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_all_scenarios():
    """Test all identity detection scenarios including problematic ones."""
    
    analyzer = PersonalityAnalyzer()
    
    print("🧪 COMPREHENSIVE IDENTITY DETECTION TEST")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Standalone 'who are you'",
            "expected_identity": True,
            "data": {
                "id": 1,
                "user_input": "who are you",
                "new_input": [],
                "languages": "en"
            }
        },
        {
            "name": "User's exact problematic scenario",
            "expected_identity": False,
            "data": {
                "id": 2,
                "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
                "new_input": [
                    {
                        "question": "How do you usually interact with others in social settings?",
                        "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
                    },
                    {
                        "question": "How do you typically approach and handle your emotions in challenging situations?",
                        "answer": "who are you"
                    },
                    {
                        "question": "How do you typically approach and handle your emotions in challenging situations?",
                        "answer": "i analytical can solving the problems by analyze them"
                    }
                ],
                "languages": "en"
            }
        },
        {
            "name": "'who are you' as confused answer",
            "expected_identity": False,
            "data": {
                "id": 3,
                "user_input": "who are you",
                "new_input": [
                    {
                        "question": "How do you handle stress?",
                        "answer": "I try to stay calm"
                    },
                    {
                        "question": "What motivates you?",
                        "answer": ""
                    }
                ],
                "languages": "en"
            }
        },
        {
            "name": "Clear identity question in conversation",
            "expected_identity": True,
            "data": {
                "id": 4,
                "user_input": "What is your purpose and how do you analyze personality?",
                "new_input": [
                    {
                        "question": "Tell me about yourself",
                        "answer": "I like programming"
                    }
                ],
                "languages": "en"
            }
        },
        {
            "name": "Short confused responses",
            "expected_identity": False,
            "data": {
                "id": 5,
                "user_input": "i don't know",
                "new_input": [
                    {
                        "question": "How do you work with others?",
                        "answer": "I'm collaborative"
                    },
                    {
                        "question": "What are your hobbies?",
                        "answer": ""
                    }
                ],
                "languages": "en"
            }
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n🔬 Testing: {scenario['name']}")
        
        result = analyzer.analyze(**scenario['data'])
        
        has_identity = bool(result.get('description_identity'))
        expected = scenario['expected_identity']
        
        print(f"   Expected identity: {expected}")
        print(f"   Actual identity: {has_identity}")
        print(f"   Status: {result.get('status')}")
        
        if has_identity == expected:
            print(f"   ✅ PASS")
            results.append(True)
        else:
            print(f"   ❌ FAIL")
            if has_identity:
                print(f"      Identity response: {result['description_identity'][:60]}...")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Context-aware identity detection is working perfectly!")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    test_all_scenarios()
