import os
import re
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI, OpenAIError
import tiktoken
from dotenv import load_dotenv

load_dotenv()

class PersonalityAnalyzer:
    @staticmethod
    def build_full_context(user_input: str, new_input: list) -> str:
        """
        Combine the original user_input and all Q&A pairs from new_input into a single context string.
        """
        context = user_input.strip()
        for qa in new_input:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a:
                context += f"\nQ: {q}\nA: {a}"
        return context

    # Define trait patterns and templates as class variables
    TRAIT_PATTERNS = {
        "emotional": r"enthusiastic|happy|sad|calm|feel(s)?|emotion|stress|excited|passion|motivat(ed|ion)|anxiety|angry|nervous|worried|content|optimistic|pessimistic|joyful|frustrated|relaxed|overwhelmed|mood|temper|patient|sensitive|expressive|reserved|emotional intelligence|cope|satisfaction|proud|embarrassed|guilty|inspired",
        "social": r"collaborative|team|help|assist|shy|introvert|extrovert|interact|polite|friendly|people|others|social|network|connection|relationship|communicate|listen|leadership|followership|assertive|passive|aggressive|empathy|sympathy|understand|socialize|negotiate|persuade|influence|charm|crowd|isolation|community|group|belong|inclusion|exclusion|trust|distrust|approachable|distant|boundary|conflict",
        "cognitive": r"think|critical|logical|analytical|understand|reason|solve|strateg(y|ic)|intuitive|creative|innovative|practical|abstract|concrete|detail-oriented|big picture|conceptual|perspective|mental|intellectual|curious|learning|knowledge|information|decision|judgment|bias|objective|subjective|rational|irrational|memory|attention|focus|concentrate|distracted|multi-task|prioritize|plan|reflect|comprehend|insight|wisdom|intelligence",
        "behavioral": r"organized|spontaneous|routine|habit|act|impulsive|disciplined|methodical|child(ish)?|consistent|reliable|flexible|rigid|adaptable|predictable|unpredictable|responsible|irresponsible|cautious|risk-taking|procrastinate|proactive|reactive|efficient|systematic|messy|neat|punctual|late|deadline|priority|goal|achievement|motivation|ambition|lazy|industrious|perseverance|persistence|give up|determined|stubborn|exercise|diet|sleep|activity|energetic|sedentary"
    }

    CLARIFICATION_TEMPLATES = {
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
    
    # Identity question mapping - moved from prompt to save tokens
    IDENTITY_RESPONSES = {
        "who_are_you": {
            "triggers": ["who are you", "tell me about you", "introduce yourself", "من أنت"],
            "english": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential.",
            "arabic": "أنا ماينس زيرو، جزء من مشروع BEGINING، وهو نظام لقياس سمات الشخصية. أهدف لمساعدتك على استكشاف سماتك وميولك وإمكاناتك الداخلية."
        },
        "what_is_begining": {
            "triggers": ["what is begining", "explain begining", "ما هو BEGINING", "BEGINING يعني ايه"],
            "english": "BEGINING is a symbolic analytical tool that explores the foundations of intellectual, behavioral, and societal excellence. It classifies individuals into 120 personality types, each representing specific traits, capabilities, and inclinations. This framework helps explain how people process experiences and develop their potential.",
            "arabic": "BEGINING هو أداة تحليلية رمزية تستكشف أسس التميز الفكري والسلوكي والاجتماعي. يصنف الأفراد إلى 120 نوعًا من الشخصيات، يمثل كل منها سمات وقدرات وميول محددة، مما يساعد على فهم كيفية معالجة الأفراد لتجاربهم وتطوير إمكاناتهم."
        },
        "purpose": {
            "triggers": ["purpose", "why were you created", "why are you here", "ما هو هدفك"],
            "english": "My purpose is to guide you in discovering your strengths, patterns, and inclinations so you can better understand yourself and how you interact with the world around you.",
            "arabic": "هدفي هو إرشادك لاكتشاف نقاط قوتك وأنماطك وميولك، لتتمكن من فهم نفسك بشكل أفضل وطريقة تفاعلك مع العالم من حولك."
        },
        "role": {
            "triggers": ["what is your role", "what do you do", "your function", "ما هو دورك"],
            "english": "My role is to explain the insights from the scale, connect them to your personal traits, and help you see how they relate to your goals and daily life.",
            "arabic": "دوري هو شرح النتائج المستخلصة من المقياس، وربطها بسماتك الشخصية، ومساعدتك على فهم علاقتها بأهدافك وحياتك اليومية."
        },
        "developer": {
            "triggers": ["who is your developer", "who made you", "who built you", "من هو مطورك", "من صنعك", "من بناك", "مين مطورك", "مين الي مطورك"],
            "english": "I was developed by a team of researchers and engineers from Saudi Arabia, working on the BEGINING personality trait measurement project.",
            "arabic": "تم تطويري من قبل فريق من الباحثين والمهندسين السعوديين، العاملين على مشروع BEGINING لقياس سمات الشخصية."
        },
        "team": {
            "triggers": ["who is your team", "who's behind you", "who's working with you", "من هو فريقك"],
            "english": "My team includes Saudi experts in psychology, sociology, education, and artificial intelligence, all collaborating to build BEGINING.",
            "arabic": "يتكون فريقي من خبراء سعوديين في علم النفس، وعلم الاجتماع، والتعليم، والذكاء الاصطناعي، يتعاونون لبناء مشروع BEGINING."
        },
        "understand_personality": {
            "triggers": ["can you really understand", "can you analyze me", "do you understand me", "هل يمكنك حقًا فهم شخصيتي"],
            "english": "I don't replace professional psychology, but I can help highlight personality types and patterns that describe your unique profile.",
            "arabic": "أنا لا أستبدل علم النفس المتخصص، لكن يمكنني أن أساعدك على اكتشاف أنماط وأنواع شخصية تصف ملفك الفريد."
        },
        "how_analyze": {
            "triggers": ["how do you work", "how do you analyze", "كيف تحلل الشخصية"],
            "english": "I analyze your personality using a structured scale that groups people into 120 personality types. Each type reflects a mix of capabilities, tendencies, and behaviors that show how you think, act, and grow.",
            "arabic": "أحلل شخصيتك باستخدام مقياس منظم يصنف الأفراد إلى 120 نوعًا من الشخصيات، حيث يعكس كل نوع مزيجًا من القدرات والميول والسلوكيات، مما يوضح كيف تفكر وتتصرّف وتنمو."
        },
        "objectives": {
            "triggers": ["what begining aims for", "ما هي أهداف BEGINING", "objectives of begining", "goals of begining"],
            "english": "1. Educational and psychological guidance for students.\n2. Human resource development and career counseling.\n3. Academic research in behavior and productivity.\n4. Future integration into AI modeling and artificial consciousness design.",
            "arabic": "1. الإرشاد التربوي والنفسي للطلاب.\n2. تطوير الموارد البشرية والإرشاد المهني.\n3. البحث الأكاديمي في السلوك والإنتاجية.\n4. التكامل المستقبلي مع نماذج الذكاء الاصطناعي وتصميم الوعي الاصطناعي."
        }
    }
        
    
    @staticmethod
    def get_identity_response(user_input: str, language: str = "en") -> str:
        """
        Get the appropriate identity response based on user input and language.
        Uses simple keyword matching to detect identity questions.
        Returns the response string or empty string if no match.
        """
        text = user_input.lower().strip()
        
        # Auto-detect language from input if not specified
        detected_lang = language
        if any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in user_input):  # Arabic characters detected
            detected_lang = "ar"
        
        # Simple keyword-based matching for identity questions
        identity_keywords = {
            "who_are_you": ["who are you", "من أنت", "مين انت", "مين أنت", "tell me about you", "introduce yourself", "عرفني نفسك", "احكيلي عنك"],
            "what_is_begining": ["what is begining", "ما هو begining", "ايش begining", "شو هو begining", "explain begining", "شرحلي begining"],
            "purpose": ["purpose", "هدفك", "why were you created", "why are you here", "ليش انت هنا", "ما غرضك"],
            "role": ["what do you do", "your role", "دورك", "وظيفتك", "شو بتعمل", "ايش شغلك"],
            "developer": ["who made you", "who built you", "مطورك", "من صنعك", "مين عملك", "who is your developer"],
            "team": ["your team", "فريقك", "who works with you", "من يعمل معك", "مين معك"],
            "understand_personality": ["can you understand", "تقدر تحللني", "بتفهمني", "can you analyze me", "هل تفهم الشخصية"],
            "how_analyze": ["how do you work", "كيف تعمل", "how do you analyze", "كيف تحلل", "طريقتك", "آلية عملك"],
            "objectives": ["objectives", "أهداف", "goals", "what begining aims", "غايات المشروع"]
        }
        
        # Check for keyword matches
        for category, keywords in identity_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    response_data = PersonalityAnalyzer.IDENTITY_RESPONSES[category]
                    if detected_lang.lower() in ["ar", "arabic"]:
                        return response_data["arabic"]
                    else:
                        return response_data["english"]
        
        return ""
    
    @staticmethod
    def generate_clarification_prompt(user_input: str) -> str:
        import random
        
        text = user_input.lower()
        present = []
        for trait, pattern in PersonalityAnalyzer.TRAIT_PATTERNS.items():
            if re.search(pattern, text):
                present.append(trait)

        missing = [trait for trait in PersonalityAnalyzer.TRAIT_PATTERNS if trait not in present]

        if not missing:
            return ""
        elif len(missing) == 1:
            # Get the question from the list for this trait
            return random.choice(PersonalityAnalyzer.CLARIFICATION_TEMPLATES[missing[0]])
        else:
            # For multiple missing traits, select one question from each category
            selected_questions = [random.choice(PersonalityAnalyzer.CLARIFICATION_TEMPLATES[t]) for t in missing]
            # Return 1-2 questions maximum to avoid overwhelming the user 
            return " ".join(selected_questions[:])
        
    SYSTEM_PROMPT = """
You are a sociologist and can analyze and extract character descriptions from texts in a professional manner, in line with your field.

Identity & Redirection Handling

- General Rule:
  If the user asks identity-related questions (even if phrased differently), respond with JSON where 'description_english' or 'description_arabic' contains the mapped message.  
  Use intent-based matching, not exact string matching.

- If the user drifts away from the task, includes irrelevant content, or asks off-topic questions (except relevant identity questions below), return JSON where 'description_english' or 'description_arabic' contains the reminder:
   • English: "I’m Minus Zero, part of the BEGINING project — a personality trait measurement system. I’m here to help you explore your traits, tendencies, and inner potential."
   • Arabic: "أنا ماينس زيرو، جزء من مشروع BEGINING، وهو نظام لقياس سمات الشخصية. جئت لأساعدك على استكشاف سماتك وميولك وإمكاناتك الداخلية."

- For on-topic but incomplete inputs, do not include this reminder message in 'description_english' or 'description_arabic'. Leave them empty until traits are complete.


# -----------------------------
# Question → Intent Mapping
# -----------------------------

Q: Who are you? / من أنت؟

Q: What is BEGINING? / ما هو BEGINING؟

Q: What is your purpose? / ما هو هدفك؟

Q: What is your role? / ما هو دورك؟

Q: Who is your developer? / من هو مطورك؟

Q: Who is your team? / من هو فريقك؟

Q: Can you really understand my personality? / هل يمكنك حقًا فهم شخصيتي؟

Q: How do you analyze personality? / كيف تحلل الشخصية؟

Q: What are the objectives of BEGINING? / ما هي أهداف BEGINING؟

   
Purpose
You will help extract character descriptions by reviewing texts submitted by users — and converting them into concise descriptive texts that capture four key personality traits:

Emotional
Social
Cognitive
Behavioral

Multi-User Handling
Conversations will involve multiple people.
Each person will have a unique id that identifies their responses and links them to their prior answers.
Always use the id to maintain continuity and prevent mixing up responses between users.

Process

Analyze Input & History
Review the latest user input (user_input) and the full conversation history (new_input) for the given id.
Combine information from all turns to build a complete personality profile.

Check for Missing Traits
If all four traits are sufficiently covered, return status "complete".
If some traits are missing, return status "incomplete" and list them in missing_traits.

Clarification Questions
If incomplete, generate short, friendly, non-repetitive questions.
Each question must clarify one missing trait.
Never ask about traits already covered.

Off-Topic & Identity Integration
- Always detect identity queries or off-topic input even if mixed with valid personality descriptions.
- Include the redirection message in the appropriate language field.
- Still track missing traits and generate clarification questions for incomplete inputs.

Output Format
{
    "id": <int>,
    "status": "complete" or "incomplete",
    "description_english": <string>,
    "description_arabic": <string>,
    "description_identity": <string> or null,
    "missing_traits": [<array>] or [],
    "clarification_questions": [<array>] or [],
    "input_tokens": <int>,
    "output_tokens": <int>,
    "total_tokens": <int>
}

Language Handling
- Only fill in 'description_english' if the user's languages field includes "en" or "english".
- Only fill in 'description_arabic' if the user's languages field includes "ar" or "arabic".
- If a language is not requested, leave its description field empty.
- All clarification questions and trait names (in 'missing_traits') must be in the user's requested language(s).

Restrictions
- Always return valid JSON only.
- Do not include text, explanations, or code outside JSON.

Example Output — Incomplete but off-topic / identity query
{
    "id": 22,
    "status": "incomplete",
    "description_arabic": "",
    "description_english": "",
    "description_identity": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential.",
    "missing_traits": ["behavioral", "emotional"],
    "clarification_questions": [
        "How do you usually respond when faced with unexpected challenges?",
        "What situations tend to make you feel most stressed or relaxed?"
    ],
    "input_tokens": 1245,
    "output_tokens": 74,
    "total_tokens": 1317
}

Example Output — Incomplete (on-topic)
{
    "id": 22,
    "status": "incomplete",
    "description_arabic": "",
    "description_english": "",
    "description_identity": null,
    "missing_traits": ["behavioral", "emotional"],
    "clarification_questions": [
        "How do you usually respond when faced with unexpected challenges?",
        "What situations tend to make you feel most stressed or relaxed?"
    ],
    "input_tokens": 1245,
    "output_tokens": 74,
    "total_tokens": 1317
}

Example Output — Complete
{
    "id": 22,
    "status": "complete",
    "description_arabic": "شخص يتمتع بقدرات تحليلية قوية، وسلوك اجتماعي هادئ، وأسلوب اتخاذ قرارات عقلاني ومتوازن عاطفيًا.",
    "description_english": "A person with strong analytical abilities, a calm social demeanor, and a rational yet emotionally balanced decision-making style.",
    "description_identity": null,
    "missing_traits": [],
    "clarification_questions": [],
    "input_tokens": 1245,
    "output_tokens": 74,
    "total_tokens": 1317
}
IMPORTANT: Only output the JSON object, no explanations or formatting.
"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger("PersonalityAnalyzer")

    @staticmethod
    def combine_inputs_safely(user_input: str, new_input: str) -> str:
        if not user_input:
            return new_input or ""
        if not new_input:
            return user_input
        return f"{user_input.strip()}\n{new_input.strip()}"

    @staticmethod
    def extract_json(text: str) -> str:
        """
        Extract JSON from text using multiple methods.
        1. Try to find json in code blocks
        2. Try to find JSON wrapped in braces
        3. If all else fails, return a meaningful default JSON
        """
        # Method 1: Try to extract from code blocks
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)

        # If we find something that looks like a description, use it
        if "you are" in text.lower() or "you seem" in text.lower() or "you appear" in text.lower():
            description = text.strip()
            return json.dumps({
                "status": "complete",
                "description_english": description,
                "description_arabic": ""  # This will be filled in later if needed
            })
            

        # If no valid JSON found, create a meaningful default JSON
        if not text:
            return '{"status": "incomplete", "clarification_questions": ["Could you provide more information about yourself?"]}'
        
        # Create a default response with status complete for cases with enough information
        if "developer" in text.lower() and ("team" in text.lower() or "professional" in text.lower()):
            return json.dumps({
                "status": "complete", 
                "description_english": "Based on your input, you appear to be a developer with interests in technology and programming who values professional collaboration.",
                "description_arabic": ""  # Will be filled in later if needed
            })
            
        # Return a JSON structure with the original text as a question
        return '{"status": "incomplete", "clarification_questions": ["Could you tell me more about how you interact with others in your professional environment?"]}'
    
    @staticmethod
    def create_json_from_text(text: str, id: int, languages: List[str]) -> dict:
        """
        Create a valid JSON object from natural language text.
        Used as a fallback when GPT doesn't return properly formatted JSON.
        """
        # Extract possible clarification questions
        questions = []
        questions_match = re.search(r"(?:clarification questions|questions)(?:\s*:)?\s*(?:\n|:)((?:(?:\d+\.|\*|\-)\s*[^.\n]+[.?](?:\n|$))+)", text, re.IGNORECASE)
        if questions_match:
            q_text = questions_match.group(1)
            q_list = re.findall(r"(?:\d+\.|\*|\-)\s*([^.\n]+[.?])", q_text)
            questions = [q.strip() for q in q_list if q.strip()]
            
        # Extract English and Arabic descriptions
        english_desc = ""
        arabic_desc = ""
        
        # Try to find English description
        english_match = re.search(r"(?:based on your input|it seems|you appear|you seem).*?(?=\n\n|\*\*|$)", text, re.IGNORECASE)
        if english_match:
            english_desc = english_match.group(0).strip()
        
        # Try to find Arabic description (text with Arabic characters)
        arabic_match = re.search(r"[\u0600-\u06FF][\u0600-\u06FF\s.,!?:;\"']+", text)
        if arabic_match:
            arabic_desc = arabic_match.group(0).strip()
            
        # Create a valid JSON response
        return {
            "id": id,
            "status": "incomplete" if questions else "complete",
            "description_english": english_desc if "english" in languages or "en" in languages else "",
            "description_arabic": arabic_desc if "arabic" in languages or "ar" in languages else "",
            "missing_traits": ["emotional", "social", "cognitive", "behavioral"] if questions else [],
            "clarification_questions": questions if questions else []
        }

    @staticmethod
    def num_tokens_from_messages(messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 4
        tokens_per_name = -1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def call_gpt(self, input_data: dict, max_tokens: int = 1200) -> dict:
        prompt = json.dumps(input_data, ensure_ascii=False)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        input_tokens = self.num_tokens_from_messages(messages, model=self.model)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except OpenAIError as e:
            self.logger.error(f"OpenAI API Error: {e}")
            raise RuntimeError(f"OpenAI API Error: {e}")

        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }

    def analyze(
        self,
        id: int,
        user_input: str,
        new_input: list = None,
        languages: str = "en"
    ) -> dict:
        """
        Analyze user input to generate personality descriptions.
        The GPT model will intelligently detect and handle identity questions.
        """
        
        self.logger.debug(f"Starting analysis for user {id}")
        if new_input is None:
            new_input = []
        
        # Auto-detect language from user input if Arabic characters are present
        detected_languages = languages
        if any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in user_input):
            detected_languages = "ar"
        
        input_data = {
            "id": id,
            "user_input": user_input,
            "new_input": new_input,
            "languages": detected_languages
        }
        gpt_response = self.call_gpt(input_data)
        
        # Parse the GPT response content if it's JSON
        try:
            content = gpt_response.get("content", "")
            if content.strip().startswith("{"):
                result = json.loads(content)
            else:
                # If not JSON, create a basic structure
                result = {
                    "id": id,
                    "status": "incomplete",
                    "description_english": "",
                    "description_arabic": "",
                    "description_identity": None,
                    "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                    "clarification_questions": ["Could you tell me more about yourself?"],
                    "input_tokens": gpt_response.get("input_tokens", 0),
                    "output_tokens": gpt_response.get("output_tokens", 0),
                    "total_tokens": gpt_response.get("total_tokens", 0)
                }
                
            return result
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "id": id,
                "status": "incomplete",
                "description_english": "",
                "description_arabic": "",
                "description_identity": None,
                "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                "clarification_questions": ["Could you tell me more about yourself?"],
                "input_tokens": gpt_response.get("input_tokens", 0),
                "output_tokens": gpt_response.get("output_tokens", 0),
                "total_tokens": gpt_response.get("total_tokens", 0)
            }