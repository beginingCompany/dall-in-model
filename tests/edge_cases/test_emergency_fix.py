"""
Test script for the emergency fix for repeated questions about approaching problems.
"""

from app.personality_analyzer import PersonalityAnalyzer
from app.emergency_question_fix import apply_emergency_fix
import json

# Apply the emergency fix
apply_emergency_fix()

# Create an analyzer instance
analyzer = PersonalityAnalyzer()

# Test case: The problematic input example
test_input = {
    "id": 663463,
    "user_input": "I am a software developer",
    "new_input": [
        {
            "question": "Thank you for sharing that you are a software developer. Could you tell me more about how you typically handle stress or excitement in your work as a software developer?",
            "answer": "When I'm feeling stressed at work, I usually take short breaks to clear my mind and approach the problem with fresh eyes. "
        },
        {
            "question": "Could you tell me more about how you approach problems and organize your work as a developer?",
            "answer": "When I'm having at work i just get out to breath for minuts and back with full mind focus"
        }
    ],
    "languages": [
        "en"
    ]
}

# Convert new_input to a string format for testing
new_input_str = ""
for qa_item in test_input["new_input"]:
    question = qa_item.get("question", "").strip()
    answer = qa_item.get("answer", "").strip()
    if question:
        new_input_str += f"Q: {question}\n"
    if answer:
        new_input_str += f"A: {answer}\n"

print(f"\nConverted new_input string:\n{new_input_str}")

# Test the analyzer with the fix
result = analyzer.analyze(
    user_input=test_input["user_input"],
    new_input=new_input_str,
    languages=test_input["languages"],
    id=test_input["id"]
)

print("Original test input:")
print(json.dumps(test_input, indent=4))
print("\nAnalyzer result:")
print(json.dumps(result, indent=4))

# Test if the problematic question is filtered out
problematic_question = "Could you tell me more about how you approach problems and organize your work as a developer?"
if problematic_question in str(result.get("clarification_questions", [])):
    print(f"\n❌ TEST FAILED: The problematic question still appears in the results!")
else:
    print(f"\n✅ TEST PASSED: The problematic question was successfully filtered out!")
    
# Show the current clarification questions
print("\nCurrent clarification questions:")
for q in result.get("clarification_questions", []):
    print(f"- {q}")
