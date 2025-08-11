"""
Test script for the personality analyzer's ability to handle unanswered questions
by using conversation history to ask related questions
"""
from app.personality_analyzer import PersonalityAnalyzer
import json

def test_unanswered_questions():
    analyzer = PersonalityAnalyzer()
    user_id = 78901
    
    print("Testing handling of unanswered questions")
    
    # Step 1: Initial basic input
    initial_input = "I am a software developer who works at a tech company"
    print(f"\nStep 1 - Initial input: '{initial_input}'")
    
    result1 = analyzer.analyze(user_input=initial_input, languages=["en"], id=user_id)
    print(f"Status: {result1.get('status')}")
    print(f"Questions: {result1.get('clarification_questions')}")
    
    # Step 2: Provide an answer that doesn't directly address the question
    if result1.get('clarification_questions'):
        question1 = result1.get('clarification_questions')[0]
        # Deliberately provide an answer that doesn't directly address the emotional question
        tangential_answer = "I've been working as a developer for about 5 years now. My current project involves building a web application for data visualization."
        
        print(f"\nStep 2 - Question: '{question1}'")
        print(f"Providing tangential answer that doesn't directly address the question: '{tangential_answer}'")
        
        # Create conversation history
        conversation = [
            {"question": question1, "answer": tangential_answer}
        ]
        
        # Build full context
        full_context = PersonalityAnalyzer.build_full_context(initial_input, conversation)
        
        # Get updated analysis
        result2 = analyzer.analyze(user_input=full_context, languages=["en"], id=user_id)
        print(f"\nStatus: {result2.get('status')}")
        print(f"New questions: {result2.get('clarification_questions')}")
        
        # Check if the new questions are different but related to the original dimension
        if result2.get('clarification_questions'):
            print("\nVerifying that the new questions are related but different from the original question")
            if question1.lower() not in [q.lower() for q in result2.get('clarification_questions')]:
                print("✓ Success: The system did not repeat the same question")
            else:
                print("✗ Error: The system repeated the same question")
        
        # Step 3: Provide another tangential answer and see if the system asks yet another related question
        if result2.get('clarification_questions'):
            question2 = result2.get('clarification_questions')[0]
            tangential_answer2 = "I typically use TypeScript and React for frontend development. I like to keep up with the latest technologies."
            
            print(f"\nStep 3 - Question: '{question2}'")
            print(f"Providing another tangential answer: '{tangential_answer2}'")
            
            # Update conversation history
            conversation.append({"question": question2, "answer": tangential_answer2})
            
            # Build updated context
            full_context = PersonalityAnalyzer.build_full_context(initial_input, conversation)
            
            # Get updated analysis
            result3 = analyzer.analyze(user_input=full_context, languages=["en"], id=user_id)
            print(f"\nStatus: {result3.get('status')}")
            print(f"Newest questions: {result3.get('clarification_questions')}")
            
            # Verify no repetition
            if result3.get('clarification_questions'):
                previous_questions = [question1.lower(), question2.lower()]
                new_questions = [q.lower() for q in result3.get('clarification_questions')]
                
                if not any(prev_q in new_q for prev_q in previous_questions for new_q in new_questions):
                    print("✓ Success: The system did not repeat previous questions")
                else:
                    print("✗ Error: The system repeated a previous question")

if __name__ == "__main__":
    test_unanswered_questions()
