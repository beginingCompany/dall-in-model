import sys
import os
import json
import re

# Add parent directory to path to allow importing the analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.personality_analyzer import PersonalityAnalyzer

def apply_fixes():
    """Apply fixes to the PersonalityAnalyzer class for question filtering."""
    
    # 1. Add the _calculate_similarity method
    def calculate_similarity(self, str1, str2):
        """
        Calculate similarity between two strings based on word overlap.
        Returns a value between 0 (no similarity) and 1 (identical).
        """
        # Tokenize both strings into sets of words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        # If either set is empty, avoid division by zero
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity coefficient
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Calculate similarity
        return len(intersection) / len(union)
    
    # Add the method to the class
    setattr(PersonalityAnalyzer, "_calculate_similarity", calculate_similarity)
    print("✅ Added _calculate_similarity method")
    
    # 2. Fix the analyze method to better extract previous questions
    original_analyze = PersonalityAnalyzer.analyze
    
    def patched_analyze(self, user_input: str, new_input: str = "", languages: list = ["english"], id: int = 1):
        """Patched analyze method with improved question filtering."""
        
        # Perform normal processing
        result = original_analyze(self, user_input, new_input, languages, id)
        
        # Better question extraction from conversation history
        combined_text = f"{user_input}\n{new_input}" if new_input else user_input
        
        # Improved regex pattern for more robust question extraction
        qa_pattern = r'Q:\s*([^\n]+?)(?:\n|$)(?:A:\s*([^\n]*?)(?:\n|$))?'
        matches = re.findall(qa_pattern, combined_text)
        
        if matches:
            previous_questions = [q.strip().lower() for q, _ in matches if q.strip()]
            print(f"Found {len(previous_questions)} previous questions in conversation")
            
            # Filter out any repeated questions
            if previous_questions and "clarification_questions" in result:
                original_questions = result["clarification_questions"]
                filtered_questions = []
                
                for new_q in original_questions:
                    # Check if this is a repeated question
                    is_repeat = False
                    new_q_lower = new_q.lower().strip()
                    
                    for prev_q in previous_questions:
                        prev_q = prev_q.lower().strip()
                        # Check various similarity measures
                        similarity = self._calculate_similarity(new_q_lower, prev_q)
                        print(f"Similarity between '{new_q_lower}' and '{prev_q}': {similarity:.2f}")
                        
                        if (new_q_lower == prev_q or  # Exact match
                            prev_q in new_q_lower or  # Previous question contained in new
                            new_q_lower in prev_q or  # New contained in previous
                            similarity > 0.4):  # Significant word overlap
                            
                            print(f"Filtering out repeated question: '{new_q}'")
                            is_repeat = True
                            break
                    
                    if not is_repeat:
                        filtered_questions.append(new_q)
                
                # Update the result
                result["clarification_questions"] = filtered_questions
        
        return result
    
    # Apply the patched method
    setattr(PersonalityAnalyzer, "analyze", patched_analyze)
    print("✅ Patched analyze method with improved question filtering")
    
    print("All fixes applied successfully!")

if __name__ == "__main__":
    apply_fixes()
    
    # Test the fixes with a simple example
    print("\nTesting the fixes...")
    
    analyzer = PersonalityAnalyzer()
    
    # Create a conversation with previous questions
    conversation = """
    I'm a software engineer specializing in machine learning.
    
    Q: How do you approach complex problems in your work?
    A: I like to break them down into smaller components and solve each one methodically.
    """
    
    # Mock the GPT response
    original_call_gpt = analyzer.call_gpt
    
    def mock_gpt_response(*args, **kwargs):
        return {
            "content": json.dumps({
                "id": 1,
                "status": "incomplete",
                "clarification_questions": [
                    "What's your approach to solving difficult technical challenges?",  # Similar to previous
                    "How do you typically interact with your team members?"  # New question
                ],
                "description_english": "",
                "description_arabic": ""
            }),
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150
        }
    
    analyzer.call_gpt = mock_gpt_response
    
    # Test with new input
    result = analyzer.analyze(
        user_input=conversation,
        new_input="I also enjoy working on open source projects.",
        languages=["english"],
        id=1
    )
    
    # Display results
    print("\nFinal clarification questions after filtering:")
    for q in result["clarification_questions"]:
        print(f"  - {q}")
    
    # Restore original method
    analyzer.call_gpt = original_call_gpt
