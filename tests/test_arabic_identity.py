from app.personality_analyzer import PersonalityAnalyzer

# Test if the Arabic identity trigger is detected
test_answer = 'من أنت'
is_trigger = PersonalityAnalyzer._is_identity_trigger(test_answer)
print(f'Arabic answer "من أنت" is identity trigger: {is_trigger}')

# Test with some variations
test_cases = [
    'من أنت',
    'من أنت؟',
    'who are you',
    'what are you',
    'I am emotional',
    'I handle emotions well'
]

for case in test_cases:
    result = PersonalityAnalyzer._is_identity_trigger(case)
    print(f'"{case}" -> Identity trigger: {result}')
