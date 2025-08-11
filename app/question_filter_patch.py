import sys
import os
import json
import re

# Add parent directory to path to allow importing the analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.personality_analyzer import PersonalityAnalyzer

# Patch the PersonalityAnalyzer class with a fix for question filtering
def patch_analyzer():
    original_analyze = PersonalityAnalyzer.analyze
    
    # Define the patched analyze method
    def patched_analyze(self, user_input: str, new_input: str = "", languages: list = ["english"], id: int = 1):
        # Call the original method to get the basic response
        result = original_analyze(self, user_input, new_input, languages, id)
        
        # Add better question filtering
        # 1. Extract questions from conversation history
        previous_questions = []
        combined_text = user_input + "\n" + new_input if new_input else user_input
        
        # Improved regex pattern for more robust question extraction
        qa_pattern = r'Q:\s*([^\n]+?)(?:\n|$)(?:A:\s*([^\n]*?)(?:\n|$))?'
        matches = re.findall(qa_pattern, combined_text)
        if matches:
            previous_questions = [q.strip().lower() for q, _ in matches if q.strip()]
            print(f"Found {len(previous_questions)} previous questions:")
            for q in previous_questions:
                print(f"  - {q}")
        
        # 2. Filter clarification questions
        if previous_questions and "clarification_questions" in result:
            original_questions = result["clarification_questions"]
            filtered_questions = []
            
            for new_q in original_questions:
                # Check if this is a repeated question
                is_repeat = False
                new_q_lower = new_q.lower().strip()
                
                for prev_q in previous_questions:
                    # Calculate similarity - simple word overlap for now
                    words1 = set(new_q_lower.split())
                    words2 = set(prev_q.split())
                    
                    # Check for substantial overlap
                    if not words1 or not words2:
                        continue
                        
                    intersection = words1.intersection(words2)
                    similarity = len(intersection) / len(words1.union(words2))
                    
                    if similarity > 0.4 or prev_q in new_q_lower or new_q_lower in prev_q:
                        print(f"Filtering out question: '{new_q}' (similar to '{prev_q}')")
                        is_repeat = True
                        break
                
                if not is_repeat:
                    filtered_questions.append(new_q)
            
            # Update the result with filtered questions
            if filtered_questions or not result["clarification_questions"]:
                result["clarification_questions"] = filtered_questions
                
            print(f"After filtering: {len(filtered_questions)} questions remain")
        
        return result
    
    # Apply the patch
    PersonalityAnalyzer.analyze = patched_analyze
    print("✅ PersonalityAnalyzer.analyze patched successfully!")

def run_demo():
    # Create a conversation with previous questions
    conversation = """
    I'm a software developer who enjoys programming in Python.
    
    Q: How do you typically interact with your colleagues?
    A: I mostly work independently but I join team meetings twice a week.
    
    Q: What's your approach to solving complex problems?
    A: I like to break them down into smaller parts and tackle each one methodically.
    """
    
    # Apply our patch
    patch_analyzer()
    
    # Create an analyzer and test it
    analyzer = PersonalityAnalyzer()
    
    # Mock the GPT response for testing purposes
    original_call_gpt = analyzer.call_gpt
    
    def mock_gpt_response(*args, **kwargs):
        return {
            "content": json.dumps({
                "id": 1,
                "status": "incomplete",
                "clarification_questions": [
                    "How do you interact with your team members?",  # Similar to a previous question
                    "Tell me about your morning routine or daily habits."  # New question
                ],
                "description_english": "",
                "description_arabic": ""
            }),
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
    
    analyzer.call_gpt = mock_gpt_response
    
    # Test the patched analyze method
    result = analyzer.analyze(
        user_input=conversation,
        new_input="I also enjoy hiking on weekends.",
        languages=["english"],
        id=1
    )
    
    # Display results
    print("\nFinal clarification questions:")
    for q in result["clarification_questions"]:
        print(f"  - {q}")
    
    # Restore original method
    analyzer.call_gpt = original_call_gpt

if __name__ == "__main__":
    run_demo()
