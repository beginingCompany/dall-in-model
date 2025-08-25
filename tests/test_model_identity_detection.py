#!/usr/bin/env python3
"""
Test script to verify that the GPT model can intelligently detect identity questions
without using regex patterns, and process them appropriately.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_model_identity_detection():
    """Test that the model can detect identity questions intelligently"""
    
    analyzer = PersonalityAnalyzer()
    
    # Test cases with various phrasings
    test_cases = [
        # English identity questions
        {"input": "who are you", "expected_type": "identity", "language": "en"},
        {"input": "tell me about yourself", "expected_type": "identity", "language": "en"},
        {"input": "what is your purpose", "expected_type": "identity", "language": "en"},
        {"input": "who made you", "expected_type": "identity", "language": "en"},
        {"input": "how do you work", "expected_type": "identity", "language": "en"},
        {"input": "what is BEGINING", "expected_type": "identity", "language": "en"},
        
        # Arabic identity questions
        {"input": "من أنت", "expected_type": "identity", "language": "ar"},
        {"input": "مين مطورك", "expected_type": "identity", "language": "ar"},
        {"input": "ما هو هدفك", "expected_type": "identity", "language": "ar"},
        {"input": "كيف تعمل", "expected_type": "identity", "language": "ar"},
        {"input": "ما هو BEGINING", "expected_type": "identity", "language": "ar"},
        
        # Personality-related questions (not identity)
        {"input": "I am happy and energetic", "expected_type": "personality", "language": "en"},
        {"input": "I work well with others", "expected_type": "personality", "language": "en"},
        {"input": "أنا شخص هادئ ومنطقي", "expected_type": "personality", "language": "ar"},
        
        # Off-topic questions
        {"input": "what's the weather like", "expected_type": "off-topic", "language": "en"},
        {"input": "tell me a joke", "expected_type": "off-topic", "language": "en"},
    ]
    
    results = []
    
    print("Testing Model Identity Detection (Without Regex)")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = analyzer.analyze(
                id=i,
                user_input=test_case["input"],
                languages=test_case["language"]
            )
            
            # Determine what the model detected
            detected_type = "unknown"
            if result.get("description_identity"):
                detected_type = "identity"
            elif result.get("status") == "incomplete" and not result.get("description_identity"):
                # Check if it's asking personality questions or giving redirection
                if any(trait in str(result.get("missing_traits", [])) for trait in ["emotional", "social", "cognitive", "behavioral"]):
                    detected_type = "personality"
                else:
                    detected_type = "off-topic"
            elif result.get("status") == "complete":
                detected_type = "personality"
            
            success = detected_type == test_case["expected_type"]
            
            results.append({
                "input": test_case["input"],
                "expected": test_case["expected_type"],
                "detected": detected_type,
                "success": success,
                "identity_response": result.get("description_identity", "None")
            })
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{i:2d}. {status} | Input: '{test_case['input'][:30]}...' | Expected: {test_case['expected_type']} | Detected: {detected_type}")
            
            if not success:
                print(f"     Identity Response: {result.get('description_identity', 'None')}")
                print(f"     Missing Traits: {result.get('missing_traits', [])}")
                print()
        
        except Exception as e:
            print(f"{i:2d}. ❌ ERROR | Input: '{test_case['input']}' | Error: {str(e)}")
            results.append({
                "input": test_case["input"],
                "expected": test_case["expected_type"],
                "detected": "error",
                "success": False,
                "identity_response": f"Error: {str(e)}"
            })
    
    # Calculate results
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    success_rate = (passed_tests / total_tests) * 100
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    # Show failed tests
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print(f"\nFailed Tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"  - '{test['input']}' | Expected: {test['expected']} | Got: {test['detected']}")
    
    print("\nModel-based identity detection test completed!")
    return success_rate >= 80  # Expect at least 80% success rate

if __name__ == "__main__":
    test_model_identity_detection()
