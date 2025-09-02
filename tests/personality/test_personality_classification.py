#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from personality_analyzer import PersonalityAnalyzer

def test_personality_vs_identity():
    """Test that personality descriptions are not misclassified as identity questions"""
    print("🔍 Testing Personality vs Identity Classification")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PersonalityAnalyzer()
    
    # User's exact problematic input
    test_input = {
        "id": 225985882206,
        "user_input": "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        "new_input": [
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            }
        ],
        "languages": "en"
    }
    
    print("📝 INPUT:")
    print(json.dumps(test_input, indent=2))
    print("-" * 60)
    
    result = analyzer.analyze(
        id=test_input["id"],
        user_input=test_input["user_input"],
        new_input=test_input["new_input"],
        languages=test_input["languages"]
    )
    
    print("📤 RESULT:")
    print(json.dumps(result, indent=2))
    print()
    
    # Analysis
    has_identity_response = result.get('description_identity') is not None
    has_personality_traits = result.get('description_english') != ""
    
    print("🔍 ANALYSIS:")
    print(f"   Identity Response: {'❌ INCORRECTLY TRIGGERED' if has_identity_response else '✅ CORRECTLY NONE'}")
    print(f"   Personality Analysis: {'✅ PRESENT' if has_personality_traits else '❌ MISSING'}")
    print(f"   Missing Traits: {result.get('missing_traits', [])}")
    print(f"   Status: {result.get('status', 'unknown')}")
    
    if not has_identity_response and (has_personality_traits or result.get('status') == 'incomplete'):
        print("\n🎉 SUCCESS: Correctly identified as personality data!")
        return True
    else:
        print("\n❌ FAILED: Still misclassifying as identity question")
        return False

if __name__ == "__main__":
    test_personality_vs_identity()
