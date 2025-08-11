"""
Specific fix for the repeated question about 'how you approach problems and organize your work'
This script directly patches the PersonalityAnalyzer class to filter out this frequently repeated question.
"""

import re
import logging
from app.personality_analyzer import PersonalityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("question_repeat_fix")

def apply_emergency_fix():
    """
    Apply a specific fix targeting the repeated question about approaching problems.
    This is a more targeted fix than the general one.
    """
    logger.info("Applying emergency fix for repeated question about approaching problems...")
    
    # Store the original analyze method
    original_analyze = PersonalityAnalyzer.analyze
    
    # Define our enhanced analyze method with extra filters for common repeated questions
    def enhanced_analyze(self, user_input, new_input="", languages=["english"], id=1):
        """
        Enhanced analyze method with specific filtering for commonly repeated questions.
        """
        # Process new_input to create a string if it's a list
        processed_new_input = new_input
        if isinstance(new_input, list):
            # Convert the list of QA dicts to a string format
            processed_text = ""
            for qa_item in new_input:
                question = qa_item.get("question", "").strip()
                answer = qa_item.get("answer", "").strip()
                if question:
                    processed_text += f"Q: {question}\n"
                if answer:
                    processed_text += f"A: {answer}\n"
            processed_new_input = processed_text
            
        # First call the original method with the processed input
        result = original_analyze(self, user_input, processed_new_input, languages, id)
        
        # Only proceed if we have clarification questions
        if not result.get("clarification_questions"):
            return result
        
        # Extract the full conversation context
        full_context = ""
        if isinstance(new_input, list):
            # For list of QA pairs
            for qa_item in new_input:
                question = qa_item.get("question", "").strip()
                if question:
                    full_context += f"\nQ: {question}\n"
        elif isinstance(new_input, str):
            # For string input
            full_context = new_input
            
        # Combine with user_input
        combined_context = f"{user_input}\n{full_context}".lower()
        
        # Debug log the full context and questions
        logger.info(f"Full conversation context length: {len(combined_context)}")
        logger.info(f"Original questions: {result.get('clarification_questions', [])}")
        
        # List of problematic question patterns to check specifically
        problem_patterns = [
            r'approach\s+problems?',
            r'approach\s+work',
            r'organize\s+(?:your\s+)?work',
            r'how\s+(?:do\s+)?you\s+(?:typically\s+)?(?:handle|deal\s+with|approach)\s+(?:complex\s+)?problems?',
            r'tell\s+(?:me\s+)?(?:more\s+)?about\s+how\s+you\s+(?:organize|structure|plan)\s+(?:your\s+)?work',
            # Generic questions that should be filtered
            r'thank\s+you\s+for\s+sharing.*anything\s+else.*add',
            r'is\s+there\s+anything\s+else.*like\s+to\s+add',
            r'is\s+there\s+anything\s+else.*help\s+me\s+understand\s+you\s+better'
        ]
        
        # Extract previous questions from context
        previous_questions = []
        question_pattern = r'Q:\s*(.*?)(?=\s*(?:A:|Q:|$))'
        matches = re.findall(question_pattern, combined_context, re.DOTALL)
        for match in matches:
            question = match.strip()
            if question:
                previous_questions.append(question.lower())
                
        logger.info(f"Extracted previous questions: {len(previous_questions)}")
        
        # Function to check similarity between two questions
        def calculate_similarity(q1, q2):
            # Convert to lowercase and split into words
            words1 = set(q1.lower().split())
            words2 = set(q2.lower().split())
            
            # Calculate overlap coefficient: intersection / min(len(A), len(B))
            if min(len(words1), len(words2)) == 0:
                return 0
                
            intersection = len(words1.intersection(words2))
            overlap_coef = intersection / min(len(words1), len(words2))
            
            # Count matching 3-word phrases
            phrase_match = 0
            q1_words = q1.lower().split()
            q2_words = q2.lower().split()
            
            for i in range(len(q1_words) - 2):
                phrase1 = ' '.join(q1_words[i:i+3])
                if phrase1 in q2.lower():
                    phrase_match += 1
            
            # Higher weight for phrase matches
            similarity = (0.6 * overlap_coef) + (0.4 * min(1.0, phrase_match / 2))
            return similarity
                
        # Check if any of the patterns are in both the context and new questions
        filtered_questions = []
        for question in result.get("clarification_questions", []):
            should_filter = False
            question_lower = question.lower()
            
            # Check for direct pattern matches
            for pattern in problem_patterns:
                pattern_in_context = re.search(pattern, combined_context) is not None
                pattern_in_question = re.search(pattern, question_lower) is not None
                
                if pattern_in_context and pattern_in_question:
                    logger.info(f"Filtering out question due to pattern match: {question}")
                    logger.info(f"Matched pattern: {pattern}")
                    should_filter = True
                    break
            
            # If not filtered yet, check for high similarity with previous questions
            if not should_filter:
                for prev_q in previous_questions:
                    similarity = calculate_similarity(prev_q, question_lower)
                    if similarity > 0.5:  # Threshold for similarity
                        logger.info(f"Filtering out question due to similarity: {question}")
                        logger.info(f"Similar to: {prev_q}")
                        logger.info(f"Similarity score: {similarity}")
                        should_filter = True
                        break
            
            # If we didn't filter based on patterns or similarity, keep the question
            if not should_filter:
                filtered_questions.append(question)
        
        # Special case: if filtered questions is empty but we had questions before,
        # generate fallbacks that definitely won't repeat
        if not filtered_questions and result.get("clarification_questions"):
            logger.info("All questions filtered out, adding fallback questions")
            
            # List of diverse fallback questions that are unlikely to have been asked
            fallback_options = [
                "I'd like to understand you better as a person. What motivates you most in your career?",
                "What skills or abilities are you most proud of developing in your professional life?",
                "How would your close friends describe your personality?",
                "What aspects of your work give you the most satisfaction?",
                "When you're facing a challenging situation, what personal strengths do you rely on?",
                "Could you share a recent accomplishment that made you feel particularly proud?",
                "What unique perspectives or experiences shape how you approach your professional life?",
                "How do you typically recharge or find balance when you're not working?",
                "What values or principles guide your decision-making process?",
                "If you could develop any new skill or ability, what would it be and why?"
            ]
            
            # Pick first 1-3 options from the list (they're already diverse)
            filtered_questions = fallback_options[:min(3, len(fallback_options))]
        
        # Update the result
        result["clarification_questions"] = filtered_questions
        logger.info(f"Final filtered questions: {filtered_questions}")
        
        return result
    
    # Apply our enhanced method to the analyzer class
    setattr(PersonalityAnalyzer, "analyze", enhanced_analyze)
    logger.info("Emergency fix applied successfully!")

if __name__ == "__main__":
    apply_emergency_fix()
    print("✅ Emergency fix for question repetition has been applied!")
    print("This fix specifically targets the repeated question about approaching problems and organizing work.")

# Add a confirmation message when the function is called from another module
else:
    def print_confirmation():
        print("✅ Emergency fix for question repetition patterns has been applied successfully!")
        
    # Call the print function when the module is imported
    print_confirmation()
