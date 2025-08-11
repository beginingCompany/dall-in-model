"""
Simple test to diagnose the issue with incomplete status but null clarification_questions
"""
from app.personality_analyzer import PersonalityAnalyzer
import json

def test_simple_response():
    analyzer = PersonalityAnalyzer()
    
    print("Testing simple responses")
    
    # Test with exact input from your example
    input_text = "I am a software developer"
    print(f"\nInput: '{input_text}'")
    
    # First analysis
    result1 = analyzer.analyze(user_input=input_text, languages=["en"], id=66346)
    print(f"Status: {result1.get('status')}")
    print(f"Has clarification_questions: {result1.get('clarification_questions') is not None}")
    if result1.get('clarification_questions'):
        print(f"Questions: {result1.get('clarification_questions')}")
    
    # If we got a question, provide a simple answer
    if result1.get('clarification_questions'):
        question = result1.get('clarification_questions')[0]
        answer = "When I'm feeling stressed at work, I usually take short breaks to clear my mind and approach the problem with fresh eyes."
        
        print(f"\nAnswering: '{question}' with '{answer}'")
        
        # Create conversation
        conversation = [
            {"question": question, "answer": answer}
        ]
        
        # Build full context
        full_context = PersonalityAnalyzer.build_full_context(input_text, conversation)
        
        # Get updated analysis
        result2 = analyzer.analyze(user_input=full_context, languages=["en"], id=66346)
        print(f"Status: {result2.get('status')}")
        print(f"Description: {result2.get('description_english') or 'None'}")
        print(f"Has clarification_questions: {result2.get('clarification_questions') is not None}")
        if result2.get('clarification_questions'):
            print(f"Questions: {result2.get('clarification_questions')}")
        print(f"Raw response: {json.dumps(result2, indent=2)}")

if __name__ == "__main__":
    test_simple_response()
