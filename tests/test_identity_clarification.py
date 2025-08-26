#!/usr/bin/env python3
"""
Test to verify that identity questions include clarification_questions in the response
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_identity_with_clarification():
    """Test that identity responses include clarification questions"""
    
    analyzer = PersonalityAnalyzer()
    
    test_cases = [
        {"input": "who are you", "language": "en"},
        {"input": "  مين مطورك  ", "language": "ar"},  # Test with spaces
        {"input": "what is your purpose", "language": "en"},
        {"input": " من أنت ", "language": "ar"},  # Test with spaces
    ]
    
    print("Testing Identity Questions with Clarification")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = analyzer.analyze(
                id=i,
                user_input=test_case["input"],
                languages=test_case["language"]
            )
            
            identity_response = result.get("description_identity")
            clarification_questions = result.get("clarification_questions", [])
            missing_traits = result.get("missing_traits", [])
            
            has_identity = "Yes" if identity_response else "No"
            has_clarification = "Yes" if clarification_questions else "No"
            
            print(f"{i}. Input: '{test_case['input']}'")
            print(f"   Language: {test_case['language']}")
            print(f"   Has Identity Response: {has_identity}")
            print(f"   Has Clarification Questions: {has_clarification}")
            print(f"   Missing Traits: {missing_traits}")
            
            if identity_response:
                print(f"   Identity: {identity_response[:80]}...")
            
            if clarification_questions:
                print(f"   Questions: {len(clarification_questions)} questions")
                for j, q in enumerate(clarification_questions[:2], 1):
                    print(f"     {j}. {q[:60]}...")
            
            # Check if both identity and clarification are present
            success = has_identity == "Yes" and (has_clarification == "Yes" or len(missing_traits) == 0)
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   Status: {status}")
            print()
            
        except Exception as e:
            print(f"{i}. ERROR with '{test_case['input']}': {str(e)}")
            print()
    
    print("Identity with clarification test completed!")

if __name__ == "__main__":
    test_identity_with_clarification()
