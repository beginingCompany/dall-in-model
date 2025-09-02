"""
Test script to verify the personality analyzer API fix
"""
import json
from app.personality_analyzer import PersonalityAnalyzer

def test_language_handling():
    """Test that the analyzer properly handles different language specifications"""
    analyzer = PersonalityAnalyzer()
    
    print("=" * 80)
    print("TESTING LANGUAGE HANDLING")
    print("=" * 80)
    
    test_cases = [
        {
            "name": "English as 'en'",
            "input": "I am an AI developer who loves Math and programming using Python.",
            "languages": ["en"]
        },
        {
            "name": "English as 'english'",
            "input": "I am an AI developer who loves Math and programming using Python.",
            "languages": ["english"]
        },
        {
            "name": "English as string 'en'",
            "input": "I am an AI developer who loves Math and programming using Python.",
            "languages": "en"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['name']}")
        result = analyzer.analyze(
            user_input=test["input"], 
            languages=test["languages"],
            id=1000 + i
        )
        
        print(f"Status: {result.get('status')}")
        print(f"English description present: {'Yes' if result.get('description_english') else 'No'}")
        print(f"English description length: {len(result.get('description_english', ''))}")
        
        # Check if clarification questions exist when needed
        if result.get('status') == 'incomplete':
            print(f"Questions: {result.get('clarification_questions')}")

def test_full_conversation():
    """Test a full conversation with multiple inputs"""
    analyzer = PersonalityAnalyzer()
    
    print("\n" + "=" * 80)
    print("TESTING FULL CONVERSATION")
    print("=" * 80)
    
    # Initial input
    user_input = "I am an AI developer who loves Math and programming using Python."
    new_inputs = []
    
    print(f"\nInitial input: {user_input}")
    
    # First analysis
    full_context = PersonalityAnalyzer.build_full_context(user_input, new_inputs)
    result1 = analyzer.analyze(user_input=full_context, languages=["en"], id=2001)
    
    print(f"Status after initial input: {result1.get('status')}")
    
    # Add new input
    new_inputs.append({
        "question": "Could you tell me more about how you interact with others in your professional environment?",
        "answer": "I interact with others in my professional environment mainly through team meetings and code reviews."
    })
    
    print(f"\nAdding conversation: {new_inputs[0]['question']} -> {new_inputs[0]['answer']}")
    
    # Second analysis with the new input
    full_context = PersonalityAnalyzer.build_full_context(user_input, new_inputs)
    result2 = analyzer.analyze(user_input=full_context, languages=["en"], id=2001)
    
    print(f"Status after conversation: {result2.get('status')}")
    print(f"Description present: {'Yes' if result2.get('description_english') else 'No'}")
    
    if result2.get('description_english'):
        print(f"\nFinal English description:\n{result2.get('description_english')[:300]}...")
    
if __name__ == "__main__":
    test_language_handling()
    test_full_conversation()
