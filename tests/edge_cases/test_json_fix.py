#!/usr/bin/env python3
"""
Test script to verify JSON parsing fix specifically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_json_parsing_fix():
    """Test that JSON parsing error is fixed when clarification questions are generated."""
    
    print("Testing JSON parsing fix...")
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Test case that should trigger clarification questions (incomplete traits)
    test_id = 123456
    user_input = "I work as a software engineer."  # Should only detect cognitive/behavioral traits
    new_input = []
    languages = "en"
    
    print(f"\nTest case designed to trigger clarification questions:")
    print(f"User Input: {user_input}")
    print(f"Expected: Should detect some traits but not all, requiring clarification")
    
    try:
        result = analyzer.analyze(
            id=test_id,
            user_input=user_input,
            new_input=new_input,
            languages=languages
        )
        
        print(f"\n=== Results ===")
        print(f"Status: {result['status']}")
        print(f"Missing traits: {result['missing_traits']}")
        print(f"Clarification questions: {result['clarification_questions']}")
        
        # Check if JSON parsing is working
        if result['clarification_questions'] and isinstance(result['clarification_questions'], list):
            if result['clarification_questions'][0] != "Could you tell me more about yourself?":
                print("✅ JSON parsing error is FIXED - specific clarification questions generated")
                print(f"   Generated question: {result['clarification_questions'][0]}")
            else:
                print("⚠️  Using fallback question - may indicate JSON issue (but error is handled)")
        else:
            print("❌ No clarification questions generated")
            
        # Test Arabic as well
        print(f"\n=== Testing Arabic JSON Parsing ===")
        result_ar = analyzer.analyze(
            id=test_id + 1,
            user_input="أنا مهندس برمجيات",  # "I am a software engineer" in Arabic
            new_input=[],
            languages="ar"
        )
        
        print(f"Arabic Status: {result_ar['status']}")
        print(f"Arabic Missing traits: {result_ar['missing_traits']}")
        print(f"Arabic Clarification questions: {result_ar['clarification_questions']}")
        
        if result_ar['clarification_questions'] and isinstance(result_ar['clarification_questions'], list):
            # Check if it contains Arabic characters
            question = result_ar['clarification_questions'][0]
            has_arabic = any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in question)
            if has_arabic and question != "هل يمكنك أن تخبرني المزيد عن نفسك؟":
                print("✅ Arabic JSON parsing is WORKING - specific Arabic questions generated")
                print(f"   Generated Arabic question: {question}")
            else:
                print("⚠️  Using Arabic fallback question")
        
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_json_parsing_fix()
