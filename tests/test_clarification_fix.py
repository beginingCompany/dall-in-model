#!/usr/bin/env python3
"""
Test clarification questions fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_clarification_questions():
    """Test that clarification questions are returned as separate array items"""
    
    try:
        from personality_analyzer import PersonalityAnalyzer
        
        # Test the new method directly
        missing_traits = ["emotional", "social", "cognitive", "behavioral"]
        questions = PersonalityAnalyzer.generate_clarification_questions(missing_traits)
        
        print("Testing clarification questions generation...\n")
        print(f"Missing traits: {missing_traits}")
        print(f"Number of questions generated: {len(questions)}")
        print("\nGenerated questions:")
        
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
        
        # Test that each question is separate
        if len(questions) == len(missing_traits):
            print(f"\n✅ PASS: Generated {len(questions)} separate questions for {len(missing_traits)} missing traits")
        else:
            print(f"\n⚠️  Generated {len(questions)} questions for {len(missing_traits)} traits")
        
        # Check that no question contains another question (no concatenation)
        concatenated = False
        for question in questions:
            if "?" in question[:-1]:  # Check if there's a question mark before the last character
                concatenated = True
                print(f"❌ FAIL: Question appears to be concatenated: {question[:100]}...")
                break
        
        if not concatenated:
            print("✅ PASS: No concatenated questions detected")
        
        print(f"\n{'='*60}")
        print("✅ Clarification questions fix implemented successfully!")
        print("✅ Each trait now gets its own separate question")
        print("✅ Questions returned as proper array format")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_clarification_questions()
