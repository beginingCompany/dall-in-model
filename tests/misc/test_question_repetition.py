import sys
import os
import json
from unittest import mock

# Add parent directory to path to allow importing the analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.personality_analyzer import PersonalityAnalyzer

def test_no_question_repetition():
    """Test that the system doesn't repeat questions in a conversation."""
    
    # Mock the OpenAI client response
    first_response_json = {
        "id": 1,
        "status": "incomplete",
        "missing_traits": ["social_interaction", "daily_habits"],
        "clarification_questions": [
            "How do you typically interact with colleagues at work?",
            "Could you share your morning routine?"
        ],
        "description_english": "",
        "description_arabic": "",
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150
    }
    
    second_response_json = {
        "id": 1,
        "status": "incomplete",
        "missing_traits": ["daily_habits"],
        "clarification_questions": [
            "How do you typically interact with colleagues at work?",  # This is a repeated question
            "What activities do you enjoy on weekends?"  # This is a new question
        ],
        "description_english": "",
        "description_arabic": "",
        "input_tokens": 150,
        "output_tokens": 70,
        "total_tokens": 220
    }
    
    # Create a mock OpenAI client
    mock_openai_client = mock.MagicMock()
    mock_response1 = mock.MagicMock()
    mock_response2 = mock.MagicMock()
    
    # Configure the mock to return different responses
    mock_response1.choices = [mock.MagicMock(message=mock.MagicMock(content=json.dumps(first_response_json)))]
    mock_response2.choices = [mock.MagicMock(message=mock.MagicMock(content=json.dumps(second_response_json)))]
    
    # Set up the mock to return different responses on consecutive calls
    mock_openai_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
    
    # Create analyzer with mock client
    analyzer = PersonalityAnalyzer()
    analyzer.client = mock_openai_client
    analyzer.num_tokens_from_messages = mock.MagicMock(return_value=100)  # Mock token counting
    
    # First analysis
    first_result = analyzer.analyze(
        user_input="I'm a software developer who enjoys coding.",
        languages=["english"],
        id=1
    )
    
    print("\nFirst analysis response:")
    print(f"Questions: {first_result['clarification_questions']}")
    
    # Second analysis - simulate a user answering just one of the questions
    conversation_history = """
Q: How do you typically interact with colleagues at work?
A: I mostly work independently but meet with colleagues for weekly standups. I prefer written communication over meetings.

Q: Could you share your morning routine?
A: 
"""
    
    second_result = analyzer.analyze(
        user_input=conversation_history,
        new_input="I'd rather not share my morning routine.",
        languages=["english"],
        id=1
    )
    
    print("\nSecond analysis response:")
    print(f"Questions: {second_result['clarification_questions']}")
    
    # Check that the repeated question was filtered out
    assert "How do you typically interact with colleagues at work?" not in second_result["clarification_questions"], \
        "Failed: The system repeated a question that was already asked"
    
    # Check that at least one question was still returned
    assert len(second_result["clarification_questions"]) > 0, \
        "Failed: No clarification questions were returned"
    
    print("\nTest passed! The system successfully avoided repeating questions.")

if __name__ == "__main__":
    test_no_question_repetition()
