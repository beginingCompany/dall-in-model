#!/usr/bin/env python3
"""
Debug what exactly is being sent to GPT to understand the input context.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.personality_analyzer import PersonalityAnalyzer
import json

def debug_gpt_input():
    """Debug the exact input being sent to GPT"""
    
    print("=== Debugging GPT Input ===")
    
    # Create analyzer but don't use OpenAI - we'll just see what would be sent
    analyzer = PersonalityAnalyzer()
    
    # The exact user scenario
    user_input = "who are you"
    new_input = [
        {
            "question": "How do you usually interact with others in social settings?",
            "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
        },
        {
            "question": "How do you typically approach and handle your emotions in challenging situations?", 
            "answer": "who are you"
        }
    ]
    
    # Simulate what would be built for GPT input
    conversation_context = []
    for qa in new_input:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if q:
            if a:
                conversation_context.append(f"Q: {q}\nA: {a}")
            else:
                conversation_context.append(f"Q: {q}\nA: [pending]")
    
    # Check what context flags would be set
    is_ongoing_conversation = conversation_context and len(conversation_context) > 0
    looks_like_identity = any(word in user_input.lower() for word in ["you", "your", "who", "what"])
    
    # Build conversation summary
    full_conversation_context = ""
    if new_input:
        for qa in new_input:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a:
                full_conversation_context += f"\nQ: {q}\nA: {a}"
    
    # Build comprehensive context for GPT (matching the actual implementation)
    context_analysis = ""
    if is_ongoing_conversation:
        # Extract traits already detected in conversation
        detected_traits = []
        for trait, pattern in analyzer.TRAIT_PATTERNS.items():
            if re.search(pattern, full_conversation_context.lower()):
                detected_traits.append(trait)
        
        context_analysis = f"""
CONVERSATION CONTEXT:
- Total exchanges: {len(conversation_context)}
- Current input appears to be: {'a confused/deflecting answer' if looks_like_identity else 'a normal personality response'}
- User was asked about: {new_input[-1].get('question', 'unknown') if new_input else 'unknown'}
- Previous valid answers show traits: {', '.join(detected_traits) if detected_traits else 'none clearly detected yet'}
"""
    
    # This is what would be sent to GPT
    input_data = {
        "id": 225985882206,
        "user_input": user_input,
        "new_input": new_input,
        "languages": "en",
        "context_flag": "mid_conversation" if (is_ongoing_conversation and looks_like_identity) else "normal",
        "conversation_summary": f"User has been answering personality questions. Previous exchanges: {len(conversation_context)}. Full context: {full_conversation_context}" if is_ongoing_conversation else "New conversation",
        "context_analysis": context_analysis,
        "full_conversation_text": analyzer.build_full_context(user_input, new_input)
    }
    
    print("INPUT DATA THAT WOULD BE SENT TO GPT:")
    print("="*50)
    print(json.dumps(input_data, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    print(f"is_ongoing_conversation: {is_ongoing_conversation}")
    print(f"looks_like_identity: {looks_like_identity}")
    print(f"context_flag: {input_data['context_flag']}")
    print(f"conversation_context length: {len(conversation_context)}")
    print(f"full_conversation_context: {repr(full_conversation_context)}")
    
    # Show what the system prompt contains (first 500 chars)
    print(f"\nSYSTEM PROMPT (first 500 chars):")
    print(analyzer.SYSTEM_PROMPT[:500] + "...")
    
    return input_data

if __name__ == "__main__":
    debug_gpt_input()
