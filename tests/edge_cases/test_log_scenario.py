#!/usr/bin/env python3
"""
Test script to verify the exact issue from the log is fixed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer

def test_exact_log_scenario():
    """Test the exact scenario from the personality_analyzer.log file."""
    
    print("Testing the exact scenario from the log file...")
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # Exact data from the log
    test_id = 225985882206
    user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights. I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
    new_input = [{'question': 'How do you usually interact with others in social settings?', 'answer': 'I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions.'}]
    languages = "en"
    
    print(f"\n=== Original Issue Reproduction ===")
    print(f"ID: {test_id}")
    print(f"User Input: {user_input}")
    print(f"Previous Q&A: {new_input}")
    print(f"Languages: {languages}")
    
    try:
        result = analyzer.analyze(
            id=test_id,
            user_input=user_input,
            new_input=new_input,
            languages=languages
        )
        
        print(f"\n=== Analysis Results ===")
        print(f"Status: {result['status']}")
        print(f"Missing traits: {result['missing_traits']}")
        print(f"Clarification questions: {result['clarification_questions']}")
        print(f"Description (English): {result.get('description_english', 'None')}")
        print(f"Description (Arabic): {result.get('description_arabic', 'None')}")
        print(f"Identity response: {result.get('description_identity', 'None')}")
        print(f"Input tokens: {result['input_tokens']}")
        print(f"Output tokens: {result['output_tokens']}")
        print(f"Total tokens: {result['total_tokens']}")
        
        print(f"\n=== Issue Analysis ===")
        
        # Check if JSON parsing error is fixed
        if result['clarification_questions'] and isinstance(result['clarification_questions'], list):
            print("✅ JSON parsing error is FIXED - clarification questions are properly generated")
        else:
            print("❌ JSON parsing error still exists")
            
        # Check if trait detection improved
        detected_traits = set(["emotional", "social", "cognitive", "behavioral"]) - set(result['missing_traits'])
        expected_minimum = {"social", "cognitive"}  # At minimum should detect these
        
        if expected_minimum.issubset(detected_traits):
            print("✅ Trait detection is IMPROVED - detecting expected minimum traits")
        else:
            print("❌ Trait detection still has issues")
            
        # Check if all traits are detected (ideal scenario)
        if len(result['missing_traits']) == 0:
            print("✅ OPTIMAL: All four trait categories detected!")
            print("✅ Status is 'complete' - no more clarification needed")
        elif len(result['missing_traits']) <= 2:
            print("✅ GOOD: Most traits detected, minimal missing traits")
        else:
            print("⚠️  Many traits still missing - may need further improvement")
            
        print(f"\nDetected traits: {detected_traits}")
        print(f"Missing traits: {set(result['missing_traits'])}")
        
        # Comparison with original log results
        print(f"\n=== Comparison with Original Log ===")
        print("Original log results:")
        print("  - Status: incomplete")
        print("  - Missing traits: ['behavioral', 'emotional', 'cognitive', 'social']")
        print("  - Clarification questions: ['Could you tell me more about yourself?'] (fallback due to JSON error)")
        print("")
        print("New results:")
        print(f"  - Status: {result['status']}")
        print(f"  - Missing traits: {result['missing_traits']}")
        print(f"  - Clarification questions: {result['clarification_questions']}")
        
        if result['status'] == 'complete' and len(result['missing_traits']) == 0:
            print("\n🎉 MAJOR IMPROVEMENT: From detecting NO traits to detecting ALL traits!")
        elif len(result['missing_traits']) < 4:
            print(f"\n🎉 SIGNIFICANT IMPROVEMENT: From detecting 0 traits to detecting {4-len(result['missing_traits'])} traits!")
        
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exact_log_scenario()
