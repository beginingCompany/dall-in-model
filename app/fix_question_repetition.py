import sys
import os
import re
import json

# Add parent directory to path to allow importing the analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.personality_analyzer import PersonalityAnalyzer

def patch_personality_analyzer():
    """
    Patch the PersonalityAnalyzer class to fix the issue with question repetition.
    This adds a more robust mechanism to detect and filter repeated questions.
    """
    
    # First, add a method to calculate similarity between questions
    def calculate_similarity(self, q1, q2):
        """Calculate semantic similarity between two questions."""
        q1_lower = q1.lower().strip()
        q2_lower = q2.lower().strip()
        
        # Direct comparison for exact matches
        if q1_lower == q2_lower:
            return 1.0
            
        # Check for key phrases that indicate the same question
        key_phrases = [
            ["approach", "problem", "solv"],
            ["interact", "colleague", "team"],
            ["routine", "habit", "schedule"],
            ["emotional", "feel", "emotion"],
            ["think", "decision", "approach"],
            ["work", "style", "method"],
            ["communicate", "talk", "discuss"]
        ]
        
        # Count how many key phrase groups appear in both questions
        same_phrase_groups = 0
        for phrase_group in key_phrases:
            q1_has_phrase = any(phrase in q1_lower for phrase in phrase_group)
            q2_has_phrase = any(phrase in q2_lower for phrase in phrase_group)
            if q1_has_phrase and q2_has_phrase:
                same_phrase_groups += 1
                
        if same_phrase_groups >= 1:
            return 0.7  # Strong similarity if they share key phrases
            
        # Direct substring match (strong signal)
        if q1_lower in q2_lower or q2_lower in q1_lower:
            return 0.8
            
        # Remove common question words and stop words
        stop_words = {"how", "what", "when", "where", "why", "do", "you", "your", "typically", 
                      "usually", "could", "would", "about", "tell", "me", "share", "describe"}
                      
        # Extract key terms
        q1_terms = set([w for w in q1_lower.split() if w not in stop_words])
        q2_terms = set([w for w in q2_lower.split() if w not in stop_words])
        
        # Calculate Jaccard similarity
        if not q1_terms or not q2_terms:
            return 0.0
            
        intersection = q1_terms.intersection(q2_terms)
        union = q1_terms.union(q2_terms)
        
        # Add a minimum similarity score if there's any meaningful overlap
        jaccard_score = len(intersection) / len(union)
        
        # Give higher weight to important terms like "problem", "approach", etc.
        important_terms = {"problem", "approach", "method", "solve", "interact", "team", "work", 
                          "colleague", "routine", "habit", "schedule", "process"}
        
        important_overlap = intersection.intersection(important_terms)
        if important_overlap:
            return max(jaccard_score, 0.5)  # Minimum 0.5 if important terms match
            
        return jaccard_score
    
    # Now, patch the analyze method
    original_analyze = PersonalityAnalyzer.analyze
    
    def patched_analyze(self, user_input, new_input="", languages=["english"], id=1):
        """
        Patched analyze method with better question filtering.
        """
        # First, call the original method to get the basic response
        result = original_analyze(self, user_input, new_input, languages, id)
        
        # Extract previous questions from the conversation
        combined_text = user_input + "\n" + new_input if new_input else user_input
        
        previous_questions = []
        # Even more improved regex for question extraction
        qa_pattern = r'Q:\s*(.*?)(?:\n\s*A:|$)'
        qa_matches = re.findall(qa_pattern, combined_text, re.IGNORECASE)
        
        if qa_matches:
            # Clean up extracted questions
            for q in qa_matches:
                # Remove any trailing newlines and whitespace
                clean_q = re.sub(r'\s*\n\s*', ' ', q).strip()
                if clean_q:
                    previous_questions.append(clean_q)
        
        # If we have new clarification questions, filter out any repeats
        if previous_questions and "clarification_questions" in result and result["clarification_questions"]:
            print(f"\nFiltering questions against {len(previous_questions)} previous questions:")
            for q in previous_questions:
                print(f"  - {q}")
            
            new_questions = result["clarification_questions"]
            filtered_questions = []
            
            for new_q in new_questions:
                # Check if this is similar to any previous question
                is_repeat = False
                
                for prev_q in previous_questions:
                    similarity = calculate_similarity(self, new_q, prev_q)
                    
                    if similarity >= 0.3:  # Lower threshold to be more aggressive in filtering
                        print(f"\nFiltered out: '{new_q}'")
                        print(f"Similar to:   '{prev_q}'")
                        print(f"Similarity:    {similarity:.2f}")
                        is_repeat = True
                        break
                
                if not is_repeat:
                    filtered_questions.append(new_q)
            
            # Update the result with filtered questions
            result["clarification_questions"] = filtered_questions
            print(f"\nAfter filtering: {len(filtered_questions)} questions remain")
        
        return result
    
    # Apply the patches
    setattr(PersonalityAnalyzer, "calculate_similarity", calculate_similarity)
    setattr(PersonalityAnalyzer, "analyze", patched_analyze)
    
    print("✅ PersonalityAnalyzer patched successfully!")

def test_patch():
    """Test the patch with a simple example."""
    # Apply the patch
    patch_personality_analyzer()
    
    # Create an analyzer
    analyzer = PersonalityAnalyzer()
    
    # Mock the call_gpt method for testing
    def mock_call_gpt(*args, **kwargs):
        return {
            "content": json.dumps({
                "id": 1,
                "status": "incomplete",
                "clarification_questions": [
                    "What approach do you typically use for solving complex problems?",  # Similar to previous
                    "How do you interact with your colleagues in a team setting?",  # New
                    "Do you prefer structured routines or flexible schedules?"  # New
                ]
            })
        }
    
    # Save the original method and replace it
    original_call_gpt = analyzer.call_gpt
    analyzer.call_gpt = mock_call_gpt
    
    # Create a conversation history with questions
    conversation = """
    I'm a software developer with 5 years of experience.
    
    Q: How do you approach complex problems in your work?
    A: I usually break them down into smaller parts and tackle each methodically.
    
    Q: What programming languages are you most comfortable with?
    A: I'm most proficient with Python and JavaScript.
    """
    
    # Test the patched analyze method
    result = analyzer.analyze(
        user_input=conversation,
        new_input="I also enjoy working on open source projects in my free time.",
        languages=["english"],
        id=1
    )
    
    # Print the final questions
    print("\nFinal clarification questions:")
    for q in result["clarification_questions"]:
        print(f"  - {q}")
    
    # Restore the original method
    analyzer.call_gpt = original_call_gpt

if __name__ == "__main__":
    test_patch()
