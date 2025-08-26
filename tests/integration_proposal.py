#!/usr/bin/env python3

"""
Integration proposal for hybrid question generation in PersonalityAnalyzer
"""

import random
from typing import List, Dict, Any, Optional

class EnhancedPersonalityAnalyzer:
    """
    Enhanced PersonalityAnalyzer with hybrid question generation
    """
    
    # Keep existing templates as reliable fallback
    CLARIFICATION_TEMPLATES = {
        "english": {
            "emotional": [
                "How do you usually feel in difficult or exciting situations? What emotions come up and how do you handle them?",
                "Can you describe how you typically respond emotionally to stress or unexpected challenges?",
                "What brings you the most joy or satisfaction in your life, and how do you express those feelings?",
                "How would your close friends describe your emotional temperament or typical mood?",
                "When you're facing a setback, what emotions typically arise and how do you manage them?"
            ],
            "social": [
                "Can you describe how you typically interact with others—do you enjoy helping, leading, or prefer to work alone?",
                "How do you typically behave in group settings versus one-on-one interactions?",
                "What role do you usually take in team projects or collaborative work environments?",
                "How would you describe your approach to building and maintaining relationships with others?",
                "In social situations, do you tend to initiate conversations or prefer others to approach you first?"
            ],
            "cognitive": [
                "What kind of thinking comes naturally to you? Are you analytical, imaginative, or more intuitive in decisions?",
                "How do you typically approach complex problems or challenging decisions?",
                "Do you prefer focusing on details or looking at the big picture when working on projects?",
                "How do you gather and process new information when learning something unfamiliar?",
                "When making important decisions, do you rely more on facts and logic or intuition and personal values?"
            ],
            "behavioral": [
                "Tell me about your habits or actions—do you prefer routines, act on impulse, or stay flexible?",
                "How organized are you in your daily life and work? Do you follow systems or adapt as you go?",
                "What does your typical day look like in terms of structure and activities?",
                "How do you approach deadlines and commitments? Are you typically early, on time, or last-minute?",
                "Do you tend to plan activities in advance or prefer to be spontaneous with your time?"
            ]
        }
    }

    def __init__(self, model: str = "gpt-3.5-turbo", enable_smart_questions: bool = False):
        """
        Initialize with option to enable smart contextual questions
        
        Args:
            model: OpenAI model to use
            enable_smart_questions: Whether to use AI-generated contextual questions
        """
        self.model = model
        self.enable_smart_questions = enable_smart_questions
        # ... rest of your existing initialization

    def generate_clarification_questions(self, missing_traits: list, language: str = "english", 
                                       conversation_history: list = None) -> list:
        """
        Enhanced question generation with smart contextual option
        
        Args:
            missing_traits: List of trait categories that need clarification
            language: Language for questions ("english" or "arabic")  
            conversation_history: Previous conversation for context (optional)
        """
        if not missing_traits:
            return []
        
        # Select one random trait to ask about (keeping your improvement)
        selected_trait = random.choice(missing_traits)
        
        # Determine language key for templates
        lang_key = "arabic" if language.lower() in ["ar", "arabic"] else "english"
        
        # SMART GENERATION: Try contextual AI question if enabled and context available
        if (self.enable_smart_questions and 
            conversation_history and 
            len(conversation_history) >= 2 and  # Has meaningful context
            hasattr(self, 'client')):
            
            try:
                contextual_question = self._generate_contextual_question(
                    selected_trait, conversation_history, language
                )
                if contextual_question:
                    return [contextual_question]
            except Exception as e:
                # Graceful fallback to templates on any error
                pass
        
        # FALLBACK: Use your existing reliable templates
        if selected_trait in self.CLARIFICATION_TEMPLATES[lang_key]:
            question = random.choice(self.CLARIFICATION_TEMPLATES[lang_key][selected_trait])
            return [question]
        
        return []

    def _generate_contextual_question(self, trait: str, conversation_history: list, language: str) -> str:
        """
        Generate AI-powered contextual question
        """
        # Build context summary from recent conversation
        context_parts = []
        for qa in conversation_history[-2:]:  # Last 2 exchanges for context
            answer = qa.get("answer", "").strip()
            if answer:
                # Extract key descriptors about the person
                context_parts.append(answer[:150])
        
        user_context = " | ".join(context_parts)
        
        trait_focus = {
            "emotional": "emotional responses, feelings, mood management, stress handling",
            "social": "social interactions, teamwork, relationships, communication style", 
            "cognitive": "thinking patterns, problem-solving approach, decision-making style",
            "behavioral": "daily habits, routines, organization, time management, lifestyle"
        }
        
        prompt = f"""Generate ONE specific question to understand this person's {trait} traits.

User context: {user_context}

Focus on: {trait_focus.get(trait, trait)}

Requirements:
- Reference their specific context/background when relevant
- Ask about {trait} traits specifically  
- Keep conversational and engaging
- One question only, no explanation
- Language: {language}

Question:"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=80
        )
        
        question = response.choices[0].message.content.strip()
        
        # Basic validation
        if question and len(question) > 20 and question.endswith("?"):
            return question
        
        return None  # Fallback to templates

# Example of how to integrate into your existing analyze method
def enhanced_analyze_method_example():
    """
    Show how to integrate into your existing analyze method
    """
    
    print("🔧 INTEGRATION EXAMPLE")
    print("=" * 30)
    
    print("""
    In your existing analyze() method, replace this line:
    
    clarification_questions = self.generate_clarification_questions(missing_traits, detected_languages)
    
    With this:
    
    clarification_questions = self.generate_clarification_questions(
        missing_traits, 
        detected_languages, 
        conversation_history=new_input  # Pass conversation context
    )
    """)
    
    print("\n🎛️ CONFIGURATION OPTIONS:")
    print("1. Production (safe): enable_smart_questions=False")
    print("   - Uses your proven templates")
    print("   - 100% reliable, fast, cost-effective")
    
    print("\n2. Enhanced (experimental): enable_smart_questions=True") 
    print("   - Uses AI for contextual questions")
    print("   - Falls back to templates on any issue")
    print("   - Better user experience but slight cost increase")
    
    print("\n3. Smart hybrid logic:")
    print("   - New users (< 2 exchanges): Use templates")
    print("   - Returning users (≥ 2 exchanges): Try AI with template fallback")

if __name__ == "__main__":
    enhanced_analyze_method_example()
