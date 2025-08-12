"""
Example script demonstrating effective inputs for personality analysis
"""
import os
import sys
import json
import requests

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Minimal input (will trigger minimal_input warning)
minimal_input = {
    "id": 1,
    "user_input": "I take short breaks to clear my mind.",
    "new_input": [],
    "languages": ["en"]
}

# Effective input with rich personality details
effective_input = {
    "id": 2,
    "user_input": """
    I'm a software engineer who loves solving complex problems. I enjoy working in teams
    and helping others understand technical concepts. I'm generally calm and patient,
    though I can get frustrated with inefficient processes. I prefer to plan ahead and
    organize my work, but I can adapt when needed. In social settings, I'm somewhat
    reserved at first but become more outgoing once I'm comfortable. I enjoy outdoor
    activities like hiking and cycling on weekends, and I read a lot of science fiction.
    """,
    "new_input": [
        {
            "question": "How do you handle stress at work?",
            "answer": "I take regular breaks and prioritize tasks. Sometimes I go for a short walk to clear my head."
        },
        {
            "question": "How do you approach learning new technologies?",
            "answer": "I enjoy diving deep into documentation and creating small projects to experiment with new tools."
        }
    ],
    "languages": ["en"]
}

def test_analysis_with_input(input_data, label):
    """Test the analyzer with the given input data"""
    print(f"\n\n=== Testing {label} ===\n")
    print("Input:")
    print(json.dumps(input_data, indent=2))
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/analyze-personality",
            json=input_data,
            timeout=30
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.text.strip():
            result = response.json()
            print("\nResponse:")
            print(json.dumps(result, indent=2))
            
            # Highlight key information
            print("\nHighlights:")
            print(f"- Status: {result.get('status')}")
            print(f"- Description: {result.get('description_english')[:100]}...")
            print(f"- Missing traits: {result.get('missing_traits', [])}")
            print(f"- Clarification questions: {len(result.get('clarification_questions', []))}")
        else:
            print("Empty response received")
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run the tests"""
    print("Testing personality analyzer with different input styles")
    
    # Test with minimal input
    test_analysis_with_input(minimal_input, "Minimal Input")
    
    # Test with effective input
    test_analysis_with_input(effective_input, "Effective Input")
    
    print("\n\nTest complete!")

if __name__ == "__main__":
    main()
