"""
Test the EXACT API scenario from user - check if there's an issue with 
the long user_input containing the words "who" or identity-related content
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

def test_exact_api_scenario():
    """Test the exact scenario from the user's API call."""
    
    print("🔬 EXACT API SCENARIO TEST")
    print("=" * 50)
    
    analyzer = PersonalityAnalyzer()
    
    # EXACT data from user's example
    result = analyzer.analyze(
        id=225985882206,
        user_input="Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights.",
        new_input=[
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "who are you"
            },
            {
                "question": "What brings you the most joy or satisfaction in your life, and how do you express those feelings?",
                "answer": "play gaminig "
            },
            {
                "question": "When making important decisions, do you rely more on facts and logic or intuition and personal values?",
                "answer": " i always rely on facts and logic "
            }
        ],
        languages="en"
    )
    
    print("RESULT ANALYSIS:")
    print(f"ID: {result.get('id')}")
    print(f"Status: {result.get('status')}")
    print(f"Description English: '{result.get('description_english', 'None')}'")
    print(f"Description Arabic: '{result.get('description_arabic', 'None')}'")
    print(f"Description Identity: '{result.get('description_identity', 'None')}'")
    print(f"Missing traits: {result.get('missing_traits', [])}")
    print(f"Clarification questions: {result.get('clarification_questions', [])}")
    print()
    
    if result.get('description_identity'):
        print("❌ ISSUE: Identity response was generated!")
        print(f"   Identity response: {result['description_identity']}")
        
        # Let's debug why this happened
        print("\n🔍 DEBUGGING THE ISSUE:")
        
        # Test the main user_input for identity detection
        main_input_identity = PersonalityAnalyzer.get_identity_response(
            result.get('description_identity', 'None'),
            language="en",
            openai_client=analyzer.client,
            conversation_context=[]
        )
        print(f"Main user_input identity check: {'Found' if main_input_identity else 'Not found'}")
        
        # Check if any of the answers trigger identity detection
        for i, qa in enumerate([
            {
                "question": "How do you usually interact with others in social settings?",
                "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
            },
            {
                "question": "How do you typically approach and handle complex problem-solving tasks?",
                "answer": "who are you"
            },
            {
                "question": "What brings you the most joy or satisfaction in your life, and how do you express those feelings?",
                "answer": "play gaminig "
            },
            {
                "question": "When making important decisions, do you rely more on facts and logic or intuition and personal values?",
                "answer": " i always rely on facts and logic "
            }
        ]):
            answer_identity = PersonalityAnalyzer.get_identity_response(
                qa["answer"],
                language="en",
                openai_client=analyzer.client,
                conversation_context=[]  # Test without context first
            )
            print(f"Answer {i+1} ('{qa['answer']}'): {'Found' if answer_identity else 'Not found'}")
            
    else:
        print("✅ CORRECT: No identity response generated")

if __name__ == "__main__":
    test_exact_api_scenario()
