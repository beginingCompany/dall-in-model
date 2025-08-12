"""
Example usage of the input_processor module.

This file demonstrates how to use the input processor functions
to handle personality analysis input data.
"""

import sys
import os

# Add the project root directory to the Python path
# This allows importing modules from the app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.input_processor import process_inputs, format_for_analysis

def example_usage():
    """Examples of using the input processor module."""
    
    # Example 1: Simple input with no Q&A pairs
    print("Example 1: Simple input")
    simple_data = {
        "id": 123,
        "user_input": "I am a software developer who likes to solve complex problems.",
        "new_input": [],
        "languages": "en"
    }
    result1 = format_for_analysis(simple_data)
    print(f"Input: {simple_data}")
    print(f"Output: {result1}\n")
    
    # Example 2: Input with Q&A pairs
    print("Example 2: Input with Q&A pairs")
    qa_data = {
        "id": 456,
        "user_input": "I am a marketing specialist.",
        "new_input": [
            {
                "question": "How do you handle tight deadlines?",
                "answer": "I prioritize tasks and focus on the most important ones first."
            },
            {
                "question": "What is your communication style?",
                "answer": "I prefer direct and clear communication."
            }
        ],
        "languages": "en"
    }
    result2 = format_for_analysis(qa_data)
    print(f"Input: {qa_data}")
    print(f"Output: {result2}")
    print("\nNote how user_input now contains:")
    print(f"1. Original input: '{qa_data['user_input']}'")
    print(f"2. Plus answers (without questions): '{result2['user_input']}'")
    print("\n")
    
    # Example 3: Input with multiple questions in one entry
    print("Example 3: Multiple questions in one entry")
    multi_question_data = {
        "id": 789,
        "user_input": "I work in healthcare.",
        "new_input": [
            {
                "question": "How do you handle stress at work? What techniques do you use to stay calm?",
                "answer": "I practice deep breathing and take short breaks."
            }
        ],
        "languages": "en"
    }
    result3 = format_for_analysis(multi_question_data)
    print(f"Input: {multi_question_data}")
    print(f"Output: {result3}")
    
    print("\nDemonstrating how multiple questions are processed:")
    print(f"Original question: '{multi_question_data['new_input'][0]['question']}'")
    print(f"Parsed into: {[qa['question'] for qa in result3['new_input']]}")
    print(f"User input now contains: '{result3['user_input']}'")
    print("\n")
    
    # Example 4: Direct use of process_inputs
    print("Example 4: Direct use of process_inputs")
    user_input = "I am a teacher."
    new_input = [
        {
            "question": "What age group do you teach?, What subjects do you specialize in?",
            "answer": "I teach high school students, primarily mathematics and physics."
        }
    ]
    processed = process_inputs(user_input, new_input)
    print(f"User Input: {user_input}")
    print(f"New Input: {new_input}")
    print(f"Processed: {processed}")

if __name__ == "__main__":
    example_usage()
