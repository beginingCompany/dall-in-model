"""
Final test with more comprehensive answers to get a complete personality profile
"""
from app.personality_analyzer import PersonalityAnalyzer
import json

def test_comprehensive_answers():
    analyzer = PersonalityAnalyzer()
    user_id = 54321
    
    print("Testing with comprehensive answers to get a complete personality profile")
    
    # Step 1: Initial basic input
    initial_input = "I am a software developer"
    print(f"\nStep 1 - Initial input: '{initial_input}'")
    
    result1 = analyzer.analyze(user_input=initial_input, languages=["en"], id=user_id)
    print(f"Status: {result1.get('status')}")
    
    # Step 2: Provide a comprehensive response covering all dimensions
    comprehensive_answer = """
    When I'm feeling stressed at work, I usually take short breaks to clear my mind and approach the problem with fresh eyes. 
    I tend to remain calm under pressure, though I can get excited when solving difficult problems. Finding elegant solutions 
    to complex challenges gives me a sense of satisfaction and pride.
    
    In terms of how I interact with others, I enjoy collaborating with team members on projects, especially during brainstorming 
    sessions. I'm comfortable working independently but find that bouncing ideas off colleagues often leads to better results. 
    I'm generally friendly and approachable, though I prefer small group interactions to large meetings.
    
    My thinking style is analytical and systematic. I like breaking down complex problems into smaller, more manageable parts. 
    I tend to think through different solutions before implementing one, considering both practical constraints and long-term 
    maintainability. I enjoy learning new technologies and approaches, and I'm always looking for ways to improve my skills.
    
    As for my work habits, I'm organized and methodical. I keep detailed to-do lists and track my progress on projects. 
    I'm punctual with deadlines and meetings, and I prefer to plan my work ahead of time rather than rushing at the last minute. 
    I maintain a tidy workspace, both physically and digitally, which helps me stay focused and efficient.
    """
    
    # Create conversation history with comprehensive answer
    if result1.get('clarification_questions'):
        question = result1.get('clarification_questions')[0]
        conversation = [
            {"question": question, "answer": comprehensive_answer}
        ]
        
        # Build full context
        full_context = PersonalityAnalyzer.build_full_context(initial_input, conversation)
        print(f"\nProviding comprehensive answer covering all personality dimensions")
        
        # Get updated analysis
        result2 = analyzer.analyze(user_input=full_context, languages=["en"], id=user_id)
        print(f"Status: {result2.get('status')}")
        print(f"Has questions: {'Yes' if result2.get('clarification_questions') else 'No'}")
        print(f"Has description: {'Yes' if result2.get('description_english') else 'No'}")
        
        if result2.get('description_english'):
            print(f"\nFinal description:\n{result2.get('description_english')}")
        elif result2.get('clarification_questions'):
            print(f"\nStill got questions: {result2.get('clarification_questions')}")

if __name__ == "__main__":
    test_comprehensive_answers()
