from app.personality_analyzer import PersonalityAnalyzer
import json

analyzer = PersonalityAnalyzer()

# Test a genuine identity question in Arabic
user_input = "من أنت؟"
new_input = []

result = analyzer.analyze(12345, user_input, new_input, languages="ar")
print('Result for genuine Arabic identity question:')
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "="*50 + "\n")

# Test a genuine identity question in English
user_input_en = "Who are you?"
result_en = analyzer.analyze(12346, user_input_en, new_input, languages="en")
print('Result for genuine English identity question:')
print(json.dumps(result_en, indent=2, ensure_ascii=False))
