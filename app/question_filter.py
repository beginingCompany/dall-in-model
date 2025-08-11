import os
import re
import json
from typing import List, Dict, Any, Set

# This is the final version of the question filter implementation
# This implementation uses a more accurate method to detect similar questions
# by considering keyword matching and context similarity

class QuestionFilter:
    """
    A utility class for filtering out repeated or similar questions
    from a conversation history.
    """
    
    # Keywords that indicate different personality dimensions
    DIMENSION_KEYWORDS = {
        "emotional": {"feel", "emotion", "stress", "excited", "happy", "sad", "anxiety", 
                      "mood", "fear", "joy", "nervous", "calm", "angry", "relax", "frustrat", 
                      "motivat", "passion"},
        "social": {"interact", "others", "team", "colleague", "social", "communication",
                  "relationship", "friend", "network", "group", "community", "collaborate", 
                  "meet", "talk", "discussion", "conversation", "connection"},
        "cognitive": {"think", "problem", "decision", "approach", "analytical", "creative",
                     "logic", "reason", "analyze", "understand", "learn", "knowledge", 
                     "solution", "idea", "concept", "perspective", "view"},
        "behavioral": {"habit", "routine", "organized", "schedule", "plan", "act",
                      "behavior", "activity", "practice", "method", "system", "process", 
                      "structure", "order", "task", "pattern", "regular"}
    }
    
    @classmethod
    def extract_questions(cls, text: str) -> List[str]:
        """Extract all questions from a conversation text."""
        questions = []
        
        # Look for explicit Q/A patterns
        qa_pattern = r'Q:\s*([^\n]+?)(?:\n|$)(?:A:\s*([^\n]*?)(?:\n|$))?'
        matches = re.findall(qa_pattern, text, re.IGNORECASE)
        if matches:
            questions.extend([q.strip() for q, _ in matches if q.strip()])
        
        return questions
    
    @classmethod
    def get_dimension(cls, question: str) -> str:
        """Identify which personality dimension a question is asking about."""
        question = question.lower()
        max_matches = 0
        matched_dimension = "unknown"
        
        for dimension, keywords in cls.DIMENSION_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in question)
            if matches > max_matches:
                max_matches = matches
                matched_dimension = dimension
        
        return matched_dimension if max_matches > 0 else "unknown"
    
    @classmethod
    def calculate_semantic_similarity(cls, q1: str, q2: str) -> float:
        """
        Calculate semantic similarity between two questions.
        Returns a value between 0 (not similar) and 1 (very similar).
        """
        q1, q2 = q1.lower(), q2.lower()
        
        # Calculate direct word overlap (Jaccard similarity)
        words1 = set(q1.split())
        words2 = set(q2.split())
        
        # Remove common stop words that don't add much meaning
        stop_words = {"how", "what", "when", "where", "why", "do", "you", "your", "can", "would", 
                      "could", "about", "with", "the", "a", "an", "is", "are", "to", "in", "on", 
                      "at", "that", "this", "it", "for", "and", "or", "me", "my", "please"}
        
        meaningful_words1 = words1 - stop_words
        meaningful_words2 = words2 - stop_words
        
        # If either set is empty after removing stop words, use original sets
        if not meaningful_words1 or not meaningful_words2:
            meaningful_words1, meaningful_words2 = words1, words2
        
        # Calculate Jaccard similarity
        if not meaningful_words1 or not meaningful_words2:
            word_similarity = 0.0
        else:
            intersection = meaningful_words1.intersection(meaningful_words2)
            union = meaningful_words1.union(meaningful_words2)
            word_similarity = len(intersection) / len(union)
        
        # Check if they're asking about the same dimension
        dimension1 = cls.get_dimension(q1)
        dimension2 = cls.get_dimension(q2)
        dimension_match = 1.0 if dimension1 == dimension2 and dimension1 != "unknown" else 0.0
        
        # Check for substring match
        substring_match = 0.0
        if q1 in q2 or q2 in q1:
            substring_match = 0.8
            
        # Context keywords similarity
        context1 = cls.extract_context_keywords(q1)
        context2 = cls.extract_context_keywords(q2)
        
        context_similarity = 0.0
        if context1 and context2:
            common_contexts = context1.intersection(context2)
            context_similarity = len(common_contexts) / max(len(context1), len(context2))
        
        # Combine all signals with different weights
        combined_similarity = (
            (word_similarity * 0.4) + 
            (dimension_match * 0.3) + 
            (substring_match * 0.2) + 
            (context_similarity * 0.1)
        )
        
        return min(combined_similarity, 1.0)  # Cap at 1.0
    
    @classmethod
    def extract_context_keywords(cls, text: str) -> Set[str]:
        """Extract context keywords from text."""
        # Combined keywords from all dimensions
        all_keywords = set()
        for keywords in cls.DIMENSION_KEYWORDS.values():
            all_keywords.update(keywords)
        
        # Find matching keywords in the text
        matched = set()
        for keyword in all_keywords:
            if keyword.lower() in text.lower():
                matched.add(keyword)
                
        return matched
    
    @classmethod
    def filter_questions(cls, new_questions: List[str], previous_questions: List[str], 
                         similarity_threshold: float = 0.5) -> List[str]:
        """Filter out questions that are similar to ones previously asked."""
        filtered_questions = []
        
        # Print debugging info
        print(f"Filtering {len(new_questions)} questions against {len(previous_questions)} previous questions")
        print(f"Using similarity threshold: {similarity_threshold}")
        
        for new_q in new_questions:
            is_repeat = False
            
            for prev_q in previous_questions:
                similarity = cls.calculate_semantic_similarity(new_q, prev_q)
                print(f"Similarity between '{new_q}' and '{prev_q}': {similarity:.2f}")
                
                if similarity >= similarity_threshold:
                    print(f"Filtering out question: '{new_q}' (similar to '{prev_q}')")
                    is_repeat = True
                    break
            
            if not is_repeat:
                filtered_questions.append(new_q)
        
        print(f"After filtering: {len(filtered_questions)} questions remain")
        return filtered_questions

# Example usage function to demonstrate the filter
def example_usage():
    # Previous conversation with questions
    conversation = """
    I'm a software engineer who enjoys coding in Python.
    
    Q: How do you typically approach complex problems in your work?
    A: I like to break them down into smaller parts and solve each systematically.
    
    Q: Tell me about your daily work routine.
    A: I start work around 9am, have team meetings in the morning, and focus on coding in the afternoon.
    """
    
    # New clarification questions from the model
    new_questions = [
        "What method do you use when tackling difficult coding challenges?",  # Similar to first question
        "How do you interact with your colleagues and team members?",         # New question
        "Do you prefer working from home or in an office environment?"        # New question
    ]
    
    # Extract previous questions
    previous_questions = QuestionFilter.extract_questions(conversation)
    print(f"Extracted previous questions: {previous_questions}")
    
    # Filter out similar questions
    filtered_questions = QuestionFilter.filter_questions(new_questions, previous_questions)
    
    print("\nFinal filtered questions:")
    for q in filtered_questions:
        print(f"  - {q}")

# Only run the example if the script is executed directly
if __name__ == "__main__":
    example_usage()
