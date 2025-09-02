#!/usr/bin/env python3
"""
Test script to verify the personality analyzer fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_trait_extraction():
    """Test trait extraction with the specific example from the log."""
    
    print("Testing personality analyzer fixes...")
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Test data from the log
    test_id = 225985882206
    user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights. I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
    new_input = [{'question': 'How do you usually interact with others in social settings?', 'answer': 'I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions.'}]
    languages = "en"
    
    print(f"\nInput: {user_input}")
    print(f"Previous Q&A: {new_input}")
    
    # Test pattern extraction first
    print("\n=== Testing Pattern Extraction ===")
    traits_pattern = analyzer._extract_traits_by_pattern(user_input)
    print(f"Pattern-based traits detected: {traits_pattern}")
    
    # Test the full analysis
    print("\n=== Testing Full Analysis ===")
    try:
        result = analyzer.analyze(
            id=test_id,
            user_input=user_input,
            new_input=new_input,
            languages=languages
        )
        
        print(f"\nResult:")
        print(f"  Status: {result['status']}")
        print(f"  Missing traits: {result['missing_traits']}")
        print(f"  Clarification questions: {result['clarification_questions']}")
        print(f"  Identity response: {result.get('description_identity', 'None')}")
        
        # Expected: should detect social, cognitive, emotional traits
        # Missing: should be minimal or just behavioral
        expected_traits = {"social", "cognitive", "emotional"}
        detected_traits = set(["emotional", "social", "cognitive", "behavioral"]) - set(result['missing_traits'])
        
        print(f"\nExpected traits: {expected_traits}")
        print(f"Detected traits: {detected_traits}")
        
        if expected_traits.issubset(detected_traits):
            print("✅ SUCCESS: Expected traits were detected!")
        else:
            print("❌ ISSUE: Some expected traits were not detected")
            print(f"Missing expected traits: {expected_traits - detected_traits}")
            
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trait_extraction()
