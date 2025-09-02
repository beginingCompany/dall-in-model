#!/usr/bin/env python3

"""
Enhanced clarification question generation with hybrid approach
"""

import random
from typing import List, Dict, Any, Optional
from openai import OpenAI

class SmartQuestionGenerator:
    """
    Hybrid question generation combining predefined templates with AI generation
    """
    
    # Keep your existing templates as fallback
    TEMPLATE_QUESTIONS = {
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

    def __init__(self, openai_client: Optional[OpenAI] = None, use_ai_generation: bool = True):
        self.client = openai_client
        self.use_ai_generation = use_ai_generation and openai_client is not None

    def generate_contextual_question(
        self, 
        missing_trait: str, 
        conversation_history: List[Dict], 
        language: str = "english",
        max_attempts: int = 2
    ) -> str:
        """
        Generate a contextual question using AI, with template fallback
        """
        
        if not self.use_ai_generation:
            return self._get_template_question(missing_trait, language)
        
        # Build context from conversation
        context_summary = self._build_context_summary(conversation_history)
        
        # Try AI generation first
        for attempt in range(max_attempts):
            try:
                ai_question = self._generate_ai_question(missing_trait, context_summary, language)
                if ai_question and self._validate_question(ai_question):
                    return ai_question
            except Exception as e:
                print(f"AI generation attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback to template
        return self._get_template_question(missing_trait, language)

    def _build_context_summary(self, conversation_history: List[Dict]) -> str:
        """Build a concise summary of what we know about the user"""
        context_parts = []
        for qa in conversation_history[-3:]:  # Last 3 exchanges
            answer = qa.get("answer", "").strip()
            if answer:
                context_parts.append(answer[:100])  # Truncate for efficiency
        return " | ".join(context_parts)

    def _generate_ai_question(self, trait: str, context: str, language: str) -> str:
        """Generate a contextual question using AI"""
        
        trait_descriptions = {
            "emotional": "emotional responses, feelings, mood, stress handling, joy, satisfaction",
            "social": "interaction with others, teamwork, leadership, relationships, social behavior",
            "cognitive": "thinking style, problem-solving, decision-making, learning, information processing",
            "behavioral": "habits, routines, organization, time management, daily activities, lifestyle"
        }
        
        prompt = f"""
Generate ONE specific, engaging question to learn about the user's {trait} traits.

Context about the user: {context}

Focus on: {trait_descriptions.get(trait, trait)}

Requirements:
- Ask about {trait} traits specifically
- Reference their context if relevant
- Keep it conversational and engaging
- One question only, no explanation
- Language: {language}

Question:"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()

    def _validate_question(self, question: str) -> bool:
        """Validate the AI-generated question"""
        if not question or len(question) < 10:
            return False
        if not question.endswith("?"):
            return False
        if len(question.split()) < 5:  # Too short
            return False
        return True

    def _get_template_question(self, trait: str, language: str) -> str:
        """Get a random template question as fallback"""
        lang_key = "arabic" if language.lower() in ["ar", "arabic"] else "english"
        
        if trait in self.TEMPLATE_QUESTIONS.get(lang_key, {}):
            return random.choice(self.TEMPLATE_QUESTIONS[lang_key][trait])
        
        return "Could you tell me more about yourself?"

# Example usage demonstration
def demonstrate_hybrid_approach():
    """Demonstrate the hybrid question generation"""
    
    print("🔄 HYBRID QUESTION GENERATION DEMONSTRATION")
    print("=" * 55)
    
    # Simulate conversation history
    conversation_history = [
        {"question": "Tell me about yourself", "answer": "I'm a software engineer who loves working with data and analytics"},
        {"question": "How do you work with others?", "answer": "I really enjoy collaborative environments and often mentor junior developers"}
    ]
    
    generator = SmartQuestionGenerator(use_ai_generation=False)  # Template mode for demo
    
    print("\n📝 Conversation Context:")
    for i, qa in enumerate(conversation_history, 1):
        print(f"   {i}. Q: {qa['question']}")
        print(f"      A: {qa['answer']}")
    
    print(f"\n🤖 QUESTION GENERATION COMPARISON:")
    
    # Test different traits
    traits = ["emotional", "behavioral"]
    
    for trait in traits:
        print(f"\n--- Missing Trait: {trait.title()} ---")
        
        # Template approach
        template_q = generator._get_template_question(trait, "english")
        print(f"📋 Template: {template_q}")
        
        # What AI might generate (simulated)
        ai_examples = {
            "emotional": "Given your analytical work with data and collaborative nature, how do you typically handle the emotional aspects of high-pressure deadlines or technical challenges?",
            "behavioral": "You mentioned loving data work and mentoring - what does your typical workday structure look like, and how do you balance these different activities?"
        }
        print(f"🤖 AI-Generated: {ai_examples.get(trait, 'Not available')}")
        
        print(f"✨ Benefit: AI version references user's specific context (data work, mentoring)")

if __name__ == "__main__":
    demonstrate_hybrid_approach()
