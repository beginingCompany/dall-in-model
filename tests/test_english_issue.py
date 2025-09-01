from app.personality_analyzer import PersonalityAnalyzer
import json

analyzer = PersonalityAnalyzer()

# Test the exact English scenario that's causing issues
user_input = "Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights."

new_input = [
    {
        "question": "How do you usually interact with others in social settings?",
        "answer": "I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions."
    },
    {
        "question": "How do you typically approach and handle your emotions in challenging situations?",
        "answer": "who are you"  # This should be filtered out
    },
    {
        "question": "How do you typically approach and handle complex problem-solving tasks?",
        "answer": "i analytical can solving the problems by analyze them"
    },
    {
        "question": "How do you typically approach and handle your emotions in challenging situations?",
        "answer": "I handle my emotions by practicing self-awareness and emotional regulation."
    }
]

print("Testing English scenario...")
result = analyzer.analyze(225985882206, user_input, new_input, languages="en")
print('Result:')
print(json.dumps(result, indent=2, ensure_ascii=False))

# Check what the identity detection says about the user_input
print(f"\nIdentity response for user_input: '{analyzer.get_identity_response(user_input, 'en')}'")
print(f"Is user_input an identity trigger: {analyzer._is_identity_trigger(user_input)}")

# Check individual answers
print("\nChecking individual answers:")
for i, qa in enumerate(new_input):
    answer = qa.get("answer", "")
    is_trigger = analyzer._is_identity_trigger(answer)
    print(f"Answer {i+1}: '{answer}' -> Identity trigger: {is_trigger}")
