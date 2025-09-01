from app.personality_analyzer import PersonalityAnalyzer
import json

analyzer = PersonalityAnalyzer()

print("=== TEST 1: Genuine Arabic identity question ===")
result1 = analyzer.analyze(
    225985882206,
    "من انت",
    [],
    languages="ar"
)
print(json.dumps(result1, indent=2, ensure_ascii=False))

print("\n=== TEST 2: Arabic conversation with identity trigger as confused answer ===")
result2 = analyzer.analyze(
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
print(json.dumps(result2, indent=2, ensure_ascii=False))

print("\n=== Analysis ===")
print(f"Test 1 - Identity response: {result1.get('description_identity') is not None}")
print(f"Test 2 - Identity response: {result2.get('description_identity') is not None}")
print(f"Test 1 - Clarification in Arabic: {'ar' in str(result1.get('clarification_questions', []))}")
print(f"Test 2 - Clarification in Arabic: {'ar' in str(result2.get('clarification_questions', []))}")
