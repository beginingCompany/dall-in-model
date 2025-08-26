from app.personality_analyzer import PersonalityAnalyzer
import json

analyzer = PersonalityAnalyzer()

print("=== Debugging Arabic Language Issue ===")

# Test the exact scenario you mentioned
result = analyzer.analyze(
    225985882206,
    "مرحبًا! أنا شخص يستمتع حقًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أجد متعة كبيرة في اكتشاف الأنماط واستخلاص الرؤى.",
    [
        {
            "question": "كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟",
            "answer": "أحب العمل ضمن فرق وغالبًا ما أجد نفسي أتولى أدوارًا قيادية بشكل طبيعي. أستمتع بتوجيه الزملاء الجدد وتيسير النقاشات الجماعية."
        },
        {
            "question": "كيف تتعامل عادةً مع عواطفك في المواقف الصعبة؟",
            "answer": "من أنت"
        }
    ],
    languages="ar"
)

print("Result:")
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n=== Checking language detection ===")
user_input = "مرحبًا! أنا شخص يستمتع حقًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أجد متعة كبيرة في اكتشاف الأنماط واستخلاص الرؤى."
has_arabic = any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in user_input)
print(f"User input has Arabic characters: {has_arabic}")
print(f"Requested languages: ar")

print("\n=== Testing clarification question generation directly ===")
missing_traits = ["emotional"]
questions = analyzer.generate_clarification_questions(missing_traits, "ar")
print(f"Generated questions for Arabic: {questions}")

questions_en = analyzer.generate_clarification_questions(missing_traits, "en")
print(f"Generated questions for English: {questions_en}")

print("\n=== Analysis ===")
clarification_questions = result.get('clarification_questions', [])
if clarification_questions:
    question = clarification_questions[0]
    has_arabic_chars = any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in question)
    print(f"Clarification question: '{question}'")
    print(f"Contains Arabic characters: {has_arabic_chars}")
    print(f"Expected: Arabic, Got: {'Arabic' if has_arabic_chars else 'English'}")
else:
    print("No clarification questions found")
