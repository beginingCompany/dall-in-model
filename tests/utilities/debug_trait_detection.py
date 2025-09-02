#!/usr/bin/env python3
"""
Debug why the system isn't detecting emotional traits properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json
import re

def debug_trait_detection():
    """Debug the trait detection process"""
    
    print("=== Debugging Trait Detection ===\n")
    analyzer = PersonalityAnalyzer()
    
    # The conversation data
    user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights."
    
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "who are you"  # This should be filtered out
        },
        {
            "question": "How do you typically approach and handle complex problem-solving tasks?",
            "answer": "i analytical can solving the problems by analyze them"
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?",
            "answer": "I handle my emotions by practicing self-awareness and emotional regulation."
        }
    ]
    
    # Build the context that the system would use for trait analysis
    print("1. CHECKING CONVERSATION CONTEXT BUILDING:")
    print("-" * 50)
    
    # Check how conversation context is built in analyze method
    conversation_context = []
    for qa in new_input:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        # The system excludes current user_input, so let's see what gets included
        if q and a and a.strip() != user_input.strip():
            conversation_context.append(f"Q: {q}\nA: {a}")
            print(f"Added to context: Q: {q[:50]}... A: {a[:50]}...")
        else:
            print(f"Excluded: Q: {q[:50]}... A: {a[:50]}...")
    
    print(f"\nConversation context length: {len(conversation_context)}")
    
    # Check trait analysis on the full context
    print(f"\n2. CHECKING TRAIT DETECTION:")
    print("-" * 50)
    
    # What the identity detection sees
    print("For IDENTITY detection, context includes:")
    for qa in new_input:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if q and a and a.strip() != user_input.strip():
            print(f"  '{a}' - length: {len(a.split())} words")
    
    # What trait analysis sees  
    print(f"\nFor TRAIT analysis, full context would be:")
    full_context = ""
    for qa in new_input:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if q and a:
            full_context += f"\nQ: {q}\nA: {a}"
    
    print(f"Full context:\n{full_context}")
    
    print(f"\n3. MANUAL TRAIT DETECTION ON FULL CONTEXT:")
    print("-" * 50)
    
    for trait, pattern in analyzer.TRAIT_PATTERNS.items():
        matches = re.findall(pattern, full_context.lower())
        if matches:
            print(f"✅ {trait}: {matches}")
        else:
            print(f"❌ {trait}: no matches")
    
    print(f"\n4. TESTING THE IDENTITY FILTERING:")
    print("-" * 50)
    
    # Test if "who are you" gets filtered
    who_are_you_context = [
        "Q: How do you usually interact with others in social settings?\nA: I love working in teams and often find myself naturally taking on leadership roles."
    ]
    
    identity_response = analyzer.get_identity_response(
        "who are you",
        language="en",
        openai_client=None,
        conversation_context=who_are_you_context
    )
    
    print(f"'who are you' in conversation context: {'FILTERED' if not identity_response else 'NOT FILTERED'}")
    if identity_response:
        print(f"  Response: {identity_response[:50]}...")

if __name__ == "__main__":
    debug_trait_detection()
