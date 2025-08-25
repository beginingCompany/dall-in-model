import os
os.environ["OPENAI_API_KEY"] = "test-key"  # Just for testing

from app.personality_analyzer import PersonalityAnalyzer

# Test the identity detection
result = PersonalityAnalyzer.get_identity_response('Who is your developer?', 'en')
print(f'Identity Result: "{result}"')
print(f'Length: {len(result)}')

# Test analyze method
analyzer = PersonalityAnalyzer()
try:
    # Mock the call_gpt method to return a simple response
    def mock_call_gpt(self, input_data, max_tokens=1200):
        return {
            "content": '{"id": 123, "status": "incomplete", "description_english": "", "description_arabic": "", "missing_traits": ["emotional"], "clarification_questions": ["test"]}',
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
    
    # Replace the method temporarily
    analyzer.call_gpt = mock_call_gpt.__get__(analyzer, PersonalityAnalyzer)
    
    analyze_result = analyzer.analyze(
        id=123,
        user_input="Who is your developer?",
        new_input=[],
        languages="en"
    )
    
    print(f'\nAnalyze Result: {analyze_result}')
    print(f'Description Identity: "{analyze_result.get("description_identity")}"')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
