from app.personality_analyzer import PersonalityAnalyzer
import json
import time

def test_personality_analyzer():
    print("Testing updated personality analyzer...")
    
    analyzer = PersonalityAnalyzer()
    
    # Test 1: Simple input with minimal information
    print("\nTest 1: Simple input with minimal information")
    result = analyzer.analyze('I am a software developer', languages=['en'])
    print(f"Status: {result.get('status')}")
    print(f"Has questions: {'Yes' if result.get('clarification_questions') else 'No'}")
    print(f"Description: {result.get('description_english')[:100]}..." if result.get('description_english') else "No description")
    
    # Test 2: More detailed input covering multiple dimensions
    print("\nTest 2: More detailed input covering multiple dimensions")
    detailed_input = """
    I'm a software developer with 5 years of experience. I enjoy solving complex problems
    and working with my team to create innovative solutions. I'm usually calm under pressure
    but can get excited when discovering a new solution. I prefer to plan my work carefully
    and stick to schedules.
    """
    result = analyzer.analyze(detailed_input, languages=['en'])
    print(f"Status: {result.get('status')}")
    print(f"Has questions: {'Yes' if result.get('clarification_questions') else 'No'}")
    print(f"Description: {result.get('description_english')[:100]}..." if result.get('description_english') else "No description")
    
    # Test 3: Input in conversation format (Q&A)
    print("\nTest 3: Input in conversation format")
    conv_input = """
    I'm a software developer.
    
    Q: How do you handle stress at work?
    A: I usually take deep breaths and break down the problem into smaller parts.
    
    Q: Do you prefer working alone or in teams?
    A: I enjoy collaborating with others, especially during brainstorming sessions.
    """
    result = analyzer.analyze(conv_input, languages=['en'])
    print(f"Status: {result.get('status')}")
    print(f"Has questions: {'Yes' if result.get('clarification_questions') else 'No'}")
    print(f"Description: {result.get('description_english')[:100]}..." if result.get('description_english') else "No description")

if __name__ == "__main__":
    test_personality_analyzer()
