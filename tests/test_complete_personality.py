#!/usr/bin/env python3
"""
Test with a more complete personality conversation to see full description generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def test_complete_personality():
    """Test with comprehensive personality data to see complete description"""
    
    print("=== Testing Complete Personality Description ===")
    analyzer = PersonalityAnalyzer()
    
    # Rich personality data covering all traits
    user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights."
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions. I'm very collaborative and social."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "I stay calm and logical under pressure. I acknowledge my emotions but don't let them overwhelm me. I'm generally optimistic and handle stress well."
        },
        {
            "question": "How do you typically approach and handle complex problem-solving tasks?", 
            "answer": "I analyze problems systematically, think critically about different approaches, and focus on understanding the big picture while paying attention to important details."
        },
        {
            "question": "Tell me about your work habits and daily routines?",
            "answer": "I'm very organized and disciplined. I follow structured routines, meet deadlines consistently, and plan my activities in advance. I'm reliable and methodical in my approach."
        }
    ]
    
    print(f"User Input: '{user_input}'")
    print(f"Conversation: {len(new_input)} comprehensive Q&A exchanges")
    print("Expected: Complete status with full personality description")
    
    result = analyzer.analyze(
        id=225985882206,
        user_input=user_input,
        new_input=new_input,
        languages="en"
    )
    
    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Analysis
    print(f"\n{'='*50}")
    print("ANALYSIS:")
    
    if result.get("status") == "complete":
        print(f"✅ Status: Complete")
        
        if result.get("description_english") and len(result["description_english"]) > 50:
            print(f"✅ PERFECT: Full personality description provided")
            print(f"   Description: {result['description_english']}")
        else:
            print(f"❌ PROBLEM: Complete but no personality description!")
            print(f"   description_english: '{result.get('description_english')}'")
            
        if result.get("description_identity"):
            print(f"❌ PROBLEM: Identity response in complete analysis: {result['description_identity']}")
        else:
            print(f"✅ GOOD: No identity response (correct)")
            
    else:
        print(f"⚠️ Status: {result.get('status')} - Missing: {result.get('missing_traits')}")
        
    return result

if __name__ == "__main__":
    test_complete_personality()
