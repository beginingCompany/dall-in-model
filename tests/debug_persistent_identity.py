"""
Debug the issue where identity response persists even after user provides proper answer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def debug_persistent_identity_issue():
    """Debug why identity response persists after proper answer."""
    
    print("🐛 DEBUG: Persistent Identity Response Issue")
    print("=" * 60)
    
    analyzer = PersonalityAnalyzer()
    
    # User's exact scenario
    result = analyzer.analyze(
        id=225985882206,
        user_input="Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        new_input=[
            {
                "question": "How do you usually interact with others in group settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "who are you"
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "i analytical can solving the problems by analyze them "
            }
        ],
        languages="en"
    )
    
    print("CONVERSATION ANALYSIS:")
    print("1. Initial description: 'Hello! I'm someone who really enjoys...'")
    print("2. Group interaction answer: 'I love working in teams...'")
    print("3. Problem-solving answer #1: 'who are you' (confused)")
    print("4. Problem-solving answer #2: 'i analytical can solving...' (proper)")
    print()
    
    print("RESULT:")
    print(f"Has identity response: {bool(result.get('description_identity'))}")
    if result.get('description_identity'):
        print(f"Identity response: {result['description_identity']}")
        print("❌ PROBLEM: Identity response still triggered!")
    else:
        print("✅ CORRECT: No identity response")
    
    print(f"Status: {result.get('status')}")
    print(f"Missing traits: {result.get('missing_traits', [])}")
    print()
    
    # Debug what's happening with identity detection
    print("🔍 DEBUGGING IDENTITY DETECTION:")
    
    # Test main user input
    main_identity = PersonalityAnalyzer.get_identity_response(
        "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        language="en",
        openai_client=analyzer.client,
        conversation_context=["Q: test\nA: test"]  # With context
    )
    print(f"Main input identity check: {'Found' if main_identity else 'Not found'}")
    
    # Test each answer individually
    answers = [
        "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions.",
        "who are you",
        "i analytical can solving the problems by analyze them "
    ]
    
    for i, answer in enumerate(answers, 1):
        # Test without context first
        identity_no_context = PersonalityAnalyzer.get_identity_response(
            answer,
            language="en",
            openai_client=analyzer.client,
            conversation_context=[]
        )
        
        # Test with context
        identity_with_context = PersonalityAnalyzer.get_identity_response(
            answer,
            language="en", 
            openai_client=analyzer.client,
            conversation_context=["Q: Previous question\nA: Previous answer"]
        )
        
        print(f"Answer {i} ('{answer[:30]}...'): No context={'Found' if identity_no_context else 'Not found'}, With context={'Found' if identity_with_context else 'Not found'}")

if __name__ == "__main__":
    debug_persistent_identity_issue()
