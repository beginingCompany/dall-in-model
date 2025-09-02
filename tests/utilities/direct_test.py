"""
Direct test of the PersonalityAnalyzer class using a multi-step conversation
"""
from app.personality_analyzer import PersonalityAnalyzer
import json

def test_conversation_direct():
    analyzer = PersonalityAnalyzer()
    user_id = 12345
    
    print("Testing multi-step conversation with PersonalityAnalyzer directly")
    
    # Step 1: Initial basic input
    initial_input = "I am a software developer"
    print(f"\nStep 1 - Initial input: '{initial_input}'")
    
    result1 = analyzer.analyze(user_input=initial_input, languages=["en"], id=user_id)
    print(f"Status: {result1.get('status')}")
    print(f"Has questions: {'Yes' if result1.get('clarification_questions') else 'No'}")
    if result1.get('clarification_questions'):
        print(f"Questions: {result1.get('clarification_questions')}")
    
    # Step 2: Answer the first question
    if result1.get('clarification_questions'):
        question = result1.get('clarification_questions')[0]
        answer = "When I'm feeling stressed at work, I usually take short breaks to clear my mind and approach the problem with fresh eyes."
        print(f"\nStep 2 - Answering: '{question}' with '{answer}'")
        
        # Create conversation history
        conversation = [
            {"question": question, "answer": answer}
        ]
        
        # Build full context from the conversation
        full_context = PersonalityAnalyzer.build_full_context(initial_input, conversation)
        print(f"Full context: {full_context[:100]}...")
        
        # Get updated analysis
        result2 = analyzer.analyze(user_input=full_context, languages=["en"], id=user_id)
        print(f"Status: {result2.get('status')}")
        print(f"Has questions: {'Yes' if result2.get('clarification_questions') else 'No'}")
        print(f"Has description: {'Yes' if result2.get('description_english') else 'No'}")
        
        if result2.get('description_english'):
            print(f"\nDescription: {result2.get('description_english')[:200]}...")
        elif result2.get('clarification_questions'):
            print(f"\nFollow-up questions: {result2.get('clarification_questions')}")
            
            # Step 3: Answer the second question
            question2 = result2.get('clarification_questions')[0]
            answer2 = "I prefer to work in a structured environment with clear goals. I organize my tasks in a to-do list and track my progress."
            print(f"\nStep 3 - Answering: '{question2}' with '{answer2}'")
            
            # Update conversation history
            conversation.append({"question": question2, "answer": answer2})
            
            # Build updated full context
            full_context = PersonalityAnalyzer.build_full_context(initial_input, conversation)
            
            # Get final analysis
            result3 = analyzer.analyze(user_input=full_context, languages=["en"], id=user_id)
            print(f"Final status: {result3.get('status')}")
            print(f"Has description: {'Yes' if result3.get('description_english') else 'No'}")
            
            if result3.get('description_english'):
                print(f"\nFinal description: {result3.get('description_english')[:200]}...")

if __name__ == "__main__":
    test_conversation_direct()
