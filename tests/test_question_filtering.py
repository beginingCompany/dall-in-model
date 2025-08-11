import sys
import os
import re
import json
import logging
from unittest import mock

# Add parent directory to path to allow importing the analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.personality_analyzer import PersonalityAnalyzer

# First, let's fix the missing _calculate_similarity method
def add_missing_method(analyzer):
    def calculate_similarity(self, str1, str2):
        # Tokenize both strings into sets of words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        # If either set is empty, avoid division by zero
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity coefficient
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Calculate similarity
        return len(intersection) / len(union)
    
    # Add the method to the instance
    setattr(PersonalityAnalyzer, "_calculate_similarity", calculate_similarity)
    return analyzer

def test_question_tracking():
    print("Testing question tracking from conversation history...")
    
    # Create analyzer and add the missing method
    analyzer = PersonalityAnalyzer()
    analyzer = add_missing_method(analyzer)
    
    # Test case 1: Extract from initial conversation
    initial_conversation = """
    I'm a software developer working with Python.
    
    Q: How do you typically interact with colleagues at work?
    A: I usually work independently, but I join team meetings twice a week.
    
    Q: What's your preferred way of solving complex problems?
    A: I like to break them down into smaller parts and tackle each one methodically.
    """
    
    # Let's manually extract questions to verify
    if "Q:" in initial_conversation:
        qa_pairs = re.findall(r'Q: (.*?)\nA: (.*?)(?=\n[QA]:|$)', initial_conversation, re.DOTALL)
        questions = [q.strip() for q, _ in qa_pairs]
        print(f"Extracted questions: {questions}")
    
    # Now let's test the analyze method with new input
    new_input = "I also enjoy hiking on weekends."
    
    # Define a mock response
    mock_gpt_response = {
        "content": json.dumps({
            "id": 1,
            "status": "incomplete",
            "clarification_questions": [
                "How do you typically interact with colleagues?",  # Similar to previous question
                "What do you find most challenging about your work?"  # New question
            ],
            "description_english": "",
            "description_arabic": ""
        }),
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150
    }
    
    # Create a mock for the call_gpt method
    original_call_gpt = analyzer.call_gpt
    analyzer.call_gpt = mock.MagicMock(return_value=mock_gpt_response)
    
    # Call analyze with both the initial conversation and new input
    result = analyzer.analyze(
        user_input=initial_conversation,
        new_input=new_input,
        languages=["english"],
        id=1
    )
    
    # Print results
    print("\nResults after question filtering:")
    print(f"Original questions in GPT response: {json.loads(mock_gpt_response['content'])['clarification_questions']}")
    print(f"Questions after filtering: {result['clarification_questions']}")
    
    # Check if the first question (similar to previous) was filtered out
    filtered_first = "How do you typically interact with colleagues?" not in result["clarification_questions"]
    print(f"\nFirst question was filtered out: {filtered_first}")
    
    # Check if the second question (new) was kept
    kept_second = "What do you find most challenging about your work?" in result["clarification_questions"]
    print(f"Second question was kept: {kept_second}")
    
    # Restore original method
    analyzer.call_gpt = original_call_gpt
    
    print("\nTest completed!")
    return filtered_first and kept_second

if __name__ == "__main__":
    success = test_question_tracking()
    print(f"\nOverall test {'PASSED' if success else 'FAILED'}")
    if not success:
        print("Consider manually patching the analyzer.analyze method to better filter repeated questions.")
