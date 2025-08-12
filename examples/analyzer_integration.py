"""
Integration example showing how to use the input processor with the personality analyzer.
"""

import sys
import os

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.input_processor import format_for_analysis
from app.personality_analyzer import PersonalityAnalyzer

def analyze_with_processor():
    """
    Example of using the input processor with the analyzer directly.
    """
    # Initialize the analyzer
    analyzer = PersonalityAnalyzer()
    
    # Example raw input data
    raw_data = {
        "id": 78901,
        "user_input": "I am a doctor working in a busy hospital. I've been practicing for 10 years.",
        "new_input": [
            {
                "question": "How do you handle emergency situations?, What's your approach to patient care?",
                "answer": "I stay calm and methodical in emergencies. I focus on prioritizing critical cases while ensuring all patients receive proper attention."
            },
            {
                "question": "How do you manage work-life balance in such a demanding profession?",
                "answer": "I make sure to schedule time off regularly and keep strict boundaries between work hours and personal time."
            }
        ],
        "languages": "en"
    }
    
    # Process the data
    processed = format_for_analysis(raw_data)
    
    print("Raw data:")
    print(f"- User input: {raw_data['user_input']}")
    print(f"- Q&A pairs: {len(raw_data['new_input'])}")
    
    print("\nProcessed data:")
    print(f"- Combined input length: {len(processed['user_input'])}")
    print(f"- Structured Q&A pairs: {len(processed['new_input'])}")
    
    # Analyze the processed data
    try:
        analysis_result = analyzer.analyze(
            id=processed["id"],
            user_input=processed["user_input"],
            new_input=processed["new_input"],
            languages=processed["languages"]
        )
        
        print("\nAnalysis completed successfully!")
        print(f"Result type: {type(analysis_result)}")
        print("Analysis result contains the following keys:")
        if isinstance(analysis_result, dict):
            for key in analysis_result.keys():
                print(f"- {key}")
    except Exception as e:
        print(f"\nError during analysis: {e}")

if __name__ == "__main__":
    analyze_with_processor()
