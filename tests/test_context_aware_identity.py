"""
Test context-aware identity detection to prevent false triggers in conversations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer
from openai import OpenAI
import json

# Initialize the analyzer
analyzer = PersonalityAnalyzer()

def test_context_aware_identity_detection():
    """Test that identity detection considers conversation context."""
    
    print("🧪 Testing Context-Aware Identity Detection")
    print("=" * 60)
    
    # Test Case 1: "who are you" as standalone question (should trigger)
    print("\n1️⃣ Testing standalone identity question:")
    standalone_result = analyzer.analyze(
        id=1,
        user_input="who are you",
        new_input=[],  # No conversation history
        languages="en"
    )
    
    print(f"Input: 'who are you' (standalone)")
    print(f"Has identity response: {'description_identity' in standalone_result and standalone_result['description_identity']}")
    if standalone_result.get('description_identity'):
        print(f"Response: {standalone_result['description_identity'][:100]}...")
    
    # Test Case 2: "who are you" as answer in ongoing conversation (should NOT trigger)
    print("\n2️⃣ Testing identity phrase in conversation context:")
    conversation_result = analyzer.analyze(
        id=2,
        user_input="who are you",
        new_input=[
            {"question": "What do you enjoy doing in your free time?", "answer": "I like reading books and playing sports"},
            {"question": "How do you handle stressful situations?", "answer": "I try to stay calm and think logically"}
        ],
        languages="en"
    )
    
    print(f"Input: 'who are you' (after 2 conversation exchanges)")
    print(f"Has identity response: {'description_identity' in conversation_result and conversation_result['description_identity']}")
    if conversation_result.get('description_identity'):
        print(f"Response: {conversation_result['description_identity'][:100]}...")
    else:
        print("Correctly avoided false identity trigger in conversation")
    
    # Test Case 3: Clear identity question even in conversation (should trigger)
    print("\n3️⃣ Testing clear identity question in conversation:")
    clear_identity_result = analyzer.analyze(
        id=3,
        user_input="What is your purpose and how do you analyze personality?",
        new_input=[
            {"question": "Tell me about your hobbies", "answer": "I enjoy programming and music"}
        ],
        languages="en"
    )
    
    print(f"Input: 'What is your purpose and how do you analyze personality?' (in conversation)")
    print(f"Has identity response: {'description_identity' in clear_identity_result and clear_identity_result['description_identity']}")
    if clear_identity_result.get('description_identity'):
        print(f"Response: {clear_identity_result['description_identity'][:100]}...")
    
    # Test Case 4: Very short confused answer in conversation (should NOT trigger)
    print("\n4️⃣ Testing short confused answer:")
    confused_result = analyzer.analyze(
        id=4,
        user_input="what do you do",
        new_input=[
            {"question": "What are your main interests?", "answer": "I like technology and helping people"},
            {"question": "How would you describe your personality?", "answer": "I'm analytical and curious"}
        ],
        languages="en"
    )
    
    print(f"Input: 'what do you do' (short answer in ongoing conversation)")
    print(f"Has identity response: {'description_identity' in confused_result and confused_result['description_identity']}")
    if not confused_result.get('description_identity'):
        print("Correctly avoided false trigger for short confused answer")
    
    print("\n" + "=" * 60)
    print("🎯 Context-Aware Identity Detection Tests Complete!")
    
    # Summary
    standalone_triggered = bool(standalone_result.get('description_identity'))
    conversation_triggered = bool(conversation_result.get('description_identity'))
    clear_triggered = bool(clear_identity_result.get('description_identity'))
    confused_triggered = bool(confused_result.get('description_identity'))
    
    print(f"\n📊 Results Summary:")
    print(f"Standalone 'who are you': {standalone_triggered} (should be True)")
    print(f"'who are you' in conversation: {conversation_triggered} (should be False)")
    print(f"Clear identity question: {clear_triggered} (should be True)")
    print(f"Short confused answer: {confused_triggered} (should be False)")
    
    # Success criteria
    success = (standalone_triggered and not conversation_triggered and 
               clear_triggered and not confused_triggered)
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Context-aware identity detection working correctly!")
    else:
        print("\n⚠️ Some tests failed. Review the logic for edge cases.")
    
    return success

def test_direct_method():
    """Test the get_identity_response method directly with context."""
    
    print("\n🔬 Direct Method Testing")
    print("=" * 40)
    
    # Test without conversation context
    no_context = PersonalityAnalyzer.get_identity_response(
        "who are you",
        conversation_context=[]
    )
    print(f"'who are you' (no context): {'Found' if no_context else 'Not found'}")
    
    # Test with conversation context (should be more restrictive)
    with_context = PersonalityAnalyzer.get_identity_response(
        "who are you",
        conversation_context=["Q: What do you like? A: I like music", "Q: How are you? A: I'm fine"]
    )
    print(f"'who are you' (with context): {'Found' if with_context else 'Not found'}")
    
    # Test longer, clearer question with context (should still work)
    clear_with_context = PersonalityAnalyzer.get_identity_response(
        "What is your purpose and who developed you?",
        conversation_context=["Q: What do you like? A: I like music"]
    )
    print(f"Clear identity question (with context): {'Found' if clear_with_context else 'Not found'}")

if __name__ == "__main__":
    test_context_aware_identity_detection()
    test_direct_method()
