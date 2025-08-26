from app.personality_analyzer import PersonalityAnalyzer
import json

analyzer = PersonalityAnalyzer()

# Test the exact Arabic scenario where identity trigger appears as an answer
user_input = "مرحبًا! أنا شخص يستمتع حقًا بالعمل مع البيانات وحل المشكلات التحليلية المعقدة. أجد متعة كبيرة في اكتشاف الأنماط واستخلاص الرؤى."

new_input = [
    {
        "question": "كيف تتفاعل عادةً مع الآخرين في المواقف الاجتماعية؟",
        "answer": "أحب العمل ضمن فرق وغالبًا ما أجد نفسي أتولى أدوارًا قيادية بشكل طبيعي. أستمتع بتوجيه الزملاء الجدد وتيسير النقاشات الجماعية."
    },
    {
        "question": "كيف تتعامل عادةً مع عواطفك في المواقف الصعبة؟",
        "answer": "من أنت"  # This should be filtered out as identity trigger
    }
]

result = analyzer.analyze(12345, user_input, new_input, languages="ar")
print('Result for Arabic identity trigger scenario:')
print(json.dumps(result, indent=2, ensure_ascii=False))
