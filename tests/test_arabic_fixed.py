from app.personality_analyzer import PersonalityAnalyzer
import json

analyzer = PersonalityAnalyzer()

print("=== Testing Arabic language with identity trigger in conversation ===")
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
            "answer": "من أنت"  # This should be filtered out
        }
    ],
    languages="ar"
)

print("Result:")
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n=== Analysis ===")
print(f"Language detected: {result.get('description_identity') is None}")
print(f"Clarification questions language: {'Arabic' if any('ت' in str(q) for q in result.get('clarification_questions', [])) else 'English'}")
print(f"Identity response: {'None (correct - it was in conversation)' if result.get('description_identity') is None else 'Present'}")

print("\n=== Testing direct Arabic identity question ===")
result_direct = analyzer.analyze(
    225985882207,
    "من أنت؟",  # Direct identity question
    [],
    languages="ar"
)
print("Direct identity question result:")
print(json.dumps(result_direct, indent=2, ensure_ascii=False))
