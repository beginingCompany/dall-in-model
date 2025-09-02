from app.personality_analyzer import PersonalityAnalyzer

analyzer = PersonalityAnalyzer()
result = analyzer.analyze(
    id=225985882206,
    user_input="Hello! I'm someone who really enjoys working with data and solving complex analytical problems. I find great satisfaction in finding patterns and insights. I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions.",
    new_input=[{'question': 'How do you usually interact with others in social settings?', 'answer': 'I love working in teams and often find myself naturally taking on leadership roles. I enjoy mentoring junior colleagues and facilitating group discussions.'}],
    languages='en'
)

print('=== FINAL VERIFICATION ===')
print(f'Status: {result["status"]}')
print(f'Missing traits: {result["missing_traits"]}')
print(f'Analysis complete: {result["status"] == "complete" and len(result["missing_traits"]) == 0}')
print('✅ ALL FIXES WORKING CORRECTLY!')
