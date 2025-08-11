import json
import time
from app.personality_analyzer import PersonalityAnalyzer

def run_conversation_test():
    """Run a multi-turn conversation test to see if the analyzer can reach complete status"""
    print("\n" + "=" * 80)
    print("CONVERSATION SIMULATION TEST")
    print("=" * 80)
    
    # Create analyzer instance
    analyzer = PersonalityAnalyzer()
    
    # Start with basic information
    test_id = 9999
    user_input = "I'm a software developer who enjoys solving complex problems."
    new_input = []
    languages = ["english", "arabic"]  # Use full language names
    
    print("\nInitial Input: " + user_input)
    
    # First interaction
    full_context = PersonalityAnalyzer.build_full_context(user_input, new_input)
    result = analyzer.analyze(full_context, "", languages, id=test_id)
    
    print(f"\nInitial Status: {result['status']}")
    print(f"Questions generated: {len(result.get('clarification_questions', []))}")
    
    if result.get('clarification_questions'):
        print(f"First question: {result['clarification_questions'][0]}")
    
    # Maximum 5 conversation turns
    for turn in range(5):
        # If we got to complete status, break the loop
        if result['status'] == "complete":
            break
            
        # Get the clarification questions
        questions = result.get('clarification_questions', [])
        if not questions:
            print("No clarification questions generated, ending conversation.")
            break
            
        print(f"\n--- Conversation Turn {turn + 1} ---")
        print(f"Question: {questions[0]}")
        
        # Generate responses based on the question content
        response = ""
        if "social" in questions[0].lower() or "interact" in questions[0].lower() or "others" in questions[0].lower():
            response = "I enjoy working in small teams where everyone has clear responsibilities. I'm somewhat introverted but can be very engaged in discussions about topics I'm passionate about."
        elif "emotion" in questions[0].lower() or "feel" in questions[0].lower() or "stress" in questions[0].lower():
            response = "I tend to stay calm under pressure. I enjoy the satisfaction of solving difficult problems, and I find coding to be meditative. When stressed, I take breaks and go for walks."
        elif "routine" in questions[0].lower() or "habit" in questions[0].lower() or "day" in questions[0].lower():
            response = "I'm very organized and follow a consistent schedule. I wake up early, exercise, then work in focused blocks with breaks. I value efficiency and planning."
        elif "think" in questions[0].lower() or "decision" in questions[0].lower() or "solve" in questions[0].lower():
            response = "I approach problems methodically, breaking them down into smaller parts. I enjoy researching different solutions before deciding on the best approach. I'm analytical but also value creative thinking."
        else:
            response = "I enjoy reading technical books and hiking on weekends. I value continuous learning and try to expand my knowledge regularly. I'm dedicated to improving my skills and staying current with technology trends."
        
        print(f"Response: {response}")
        
        # Add to new_input
        new_input.append({"question": questions[0], "answer": response})
        
        # Re-analyze
        full_context = PersonalityAnalyzer.build_full_context(user_input, new_input)
        result = analyzer.analyze(full_context, "", languages, id=test_id)
        
        print(f"Status: {result['status']}")
        print(f"Questions: {len(result.get('clarification_questions', []))}")
    
    # Final result
    print("\n" + "=" * 80)
    print("CONVERSATION TEST RESULTS")
    print("=" * 80)
    print(f"Conversation ended with status: {result['status']}")
    print(f"Total turns: {turn + 1}")
    
    # Always display descriptions, even if empty (for debugging purposes)
    print("\nFinal English Description:")
    eng_desc = result.get('description_english', 'Not provided')
    print(eng_desc[:300] + "..." if len(eng_desc) > 300 else eng_desc)
    
    print("\nFinal Arabic Description:")
    ar_desc = result.get('description_arabic', 'Not provided')
    print(ar_desc[:300] + "..." if len(ar_desc) > 300 else ar_desc)

if __name__ == "__main__":
    run_conversation_test()
