from app.personality_analyzer import PersonalityAnalyzer
import json

analyzer = PersonalityAnalyzer()

print("=== TEST: Direct identity question as user_input ===")
result = analyzer.analyze(
    225985882206,
    "من أنت",  # Direct identity question
    [],  # No conversation history
    languages="ar"
)
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n=== Identity detection test ===")
identity_response = analyzer.get_identity_response("من أنت", "ar")
print(f"Identity response: '{identity_response}'")
print(f"Is identity trigger: {analyzer._is_identity_trigger('من أنت')}")
