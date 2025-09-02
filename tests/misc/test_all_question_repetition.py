"""
Test script for the enhanced fix that prevents all question repetition,
including the generic "Thank you for sharing..." questions.
"""

from app.personality_analyzer import PersonalityAnalyzer
from app.emergency_question_fix import apply_emergency_fix
import json

# Apply the emergency fix
apply_emergency_fix()

# Create an analyzer instance
analyzer = PersonalityAnalyzer()

# Test case 1: The problematic input with generic question
test_input_generic = {
    "id": 663464,
    "user_input": "I am a product designer who loves to create innovative solutions",
    "new_input": [
        {
            "question": "Thank you for sharing. Is there anything else about yourself you'd like to add that would help me understand you better?",
            "answer": "I've been working in the field for over 5 years and specialize in user experience design."
        }
    ],
    "languages": [
        "en"
    ]
}

# Test case 2: The problematic input with approach/organize question
test_input_approach = {
    "id": 663463,
    "user_input": "I am a software developer",
    "new_input": [
        {
            "question": "Could you tell me more about how you approach problems and organize your work as a developer?",
            "answer": "When I'm having at work i just get out to breath for minuts and back with full mind focus"
        }
    ],
    "languages": [
        "en"
    ]
}

# Convert the inputs to string format
def convert_input_to_string(input_obj):
    result = input_obj["user_input"] + "\n"
    for qa_item in input_obj["new_input"]:
        question = qa_item.get("question", "").strip()
        answer = qa_item.get("answer", "").strip()
        if question:
            result += f"Q: {question}\n"
        if answer:
            result += f"A: {answer}\n"
    return result

# Test the analyzer with generic question input
print("=== TEST 1: GENERIC QUESTION ===")
new_input_str_generic = convert_input_to_string(test_input_generic)
print(f"\nInput:\n{new_input_str_generic}")

result_generic = analyzer.analyze(
    user_input=test_input_generic["user_input"],
    new_input=new_input_str_generic,
    languages=test_input_generic["languages"],
    id=test_input_generic["id"]
)

# Test the analyzer with approach question input
print("\n=== TEST 2: APPROACH/ORGANIZE QUESTION ===")
new_input_str_approach = convert_input_to_string(test_input_approach)
print(f"\nInput:\n{new_input_str_approach}")

result_approach = analyzer.analyze(
    user_input=test_input_approach["user_input"],
    new_input=new_input_str_approach,
    languages=test_input_approach["languages"],
    id=test_input_approach["id"]
)

# Test if the problematic questions were filtered out
generic_question = "Thank you for sharing. Is there anything else about yourself you'd like to add that would help me understand you better"
approach_question = "Could you tell me more about how you approach problems and organize your work as a developer"

print("\n=== RESULTS ===")
print("\nGeneric question test:")
if generic_question.lower() in str(result_generic.get("clarification_questions", [])).lower():
    print(f"❌ TEST FAILED: The generic question still appears in the results!")
else:
    print(f"✅ TEST PASSED: The generic question was successfully filtered out!")
    
print("\nApproach question test:")
if approach_question.lower() in str(result_approach.get("clarification_questions", [])).lower():
    print(f"❌ TEST FAILED: The approach question still appears in the results!")
else:
    print(f"✅ TEST PASSED: The approach question was successfully filtered out!")
    
# Show the current clarification questions
print("\nCurrent clarification questions (Generic test):")
for q in result_generic.get("clarification_questions", []):
    print(f"- {q}")
    
print("\nCurrent clarification questions (Approach test):")
for q in result_approach.get("clarification_questions", []):
    print(f"- {q}")
