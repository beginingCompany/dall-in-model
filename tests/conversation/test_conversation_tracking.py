import sys
import os
import re
import json
import logging
from unittest import mock

# Add parent directory to path to allow importing the analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.personality_analyzer import PersonalityAnalyzer

# Configure logging to see debug messages
logging.basicConfig(level=logging.INFO)

def test_conversation_tracking():
    """Test that the system tracks questions from a conversation."""
    
    # Create a personality analyzer instance
    analyzer = PersonalityAnalyzer()
    
    # Check if the similarity calculation method is working
    similarity = analyzer._calculate_similarity("How do you interact with colleagues?", "How do you usually interact with your team members?")
    print(f"Similarity between questions: {similarity}")
    
    # Test extraction of previous questions
    conversation = """
    I'm a software developer working on AI systems.
    
    Q: How do you typically interact with colleagues at work?
    A: I mostly work independently but meet with colleagues for weekly standups.
    
    Q: Could you share your morning routine?
    A: I usually wake up at 7am, exercise, then start work around 9am.
    """
    
    # Let's add new input that contains the whole conversation 
    new_input = "What else would you like to know about me?"
    
    # Extract questions manually to check our method
    if "Q:" in conversation:
        qa_pairs = re.findall(r'Q: (.*?)\nA: (.*?)(?=\n[QA]:|$)', conversation, re.DOTALL)
        previous_questions = [q.strip() for q, _ in qa_pairs]
        print(f"Extracted questions: {previous_questions}")
    
    # Now do the same using our analyzer
    combined_text = analyzer.combine_inputs_safely(conversation, new_input)
    if "Q:" in combined_text:
        qa_pairs = re.findall(r'Q: (.*?)\nA: (.*?)(?=\n[QA]:|$)', combined_text, re.DOTALL)
        previous_questions = [q.strip().lower() for q, _ in qa_pairs]
        print(f"Previous questions from combined text: {previous_questions}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_conversation_tracking()
