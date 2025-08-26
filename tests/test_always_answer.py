from app.personality_analyzer import PersonalityAnalyzer
import json

analyzer = PersonalityAnalyzer()

print("=== TEST 1: Identity question in mid-conversation (should now work) ===")
result1 = analyzer.analyze(
    225985882206,
    "من أنت",  # Identity question as main user_input
    [
        {
            "question": "كيف تتفاعل مع الآخرين؟",
            "answer": "أحب العمل مع الفرق"
        }
    ],  # Has conversation history
    languages="ar"
)
print(json.dumps(result1, indent=2, ensure_ascii=False))

print("\n=== TEST 2: Identity question with extensive conversation history ===")
result2 = analyzer.analyze(
    225985882206,
    "Who are you?",  # Identity question as main user_input  
    [
        {
            "question": "How do you interact with others?",
            "answer": "I love working in teams and taking leadership roles."
        },
        {
            "question": "How do you handle emotions?",
            "answer": "I practice self-awareness and emotional regulation."
        },
        {
            "question": "How do you approach problems?",
            "answer": "I'm analytical and logical in my approach."
        }
    ],  # Extensive conversation history
    languages="en"
)
print(json.dumps(result2, indent=2, ensure_ascii=False))

print("\n=== Analysis ===")
print(f"Test 1 - Identity response: {'✅' if result1.get('description_identity') else '❌'}")
print(f"Test 2 - Identity response: {'✅' if result2.get('description_identity') else '❌'}")
print("\nBoth should now show ✅ - identity questions always answered!")
