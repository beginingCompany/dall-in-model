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
    def _is_identity_trigger(answer: str, openai_client=None, logger=None) -> bool:
        """
        Use GPT to detect if an answer is an identity trigger.
        """
        if logger:
            logger.info(f"_is_identity_trigger: answer={answer}")
        if not answer or len(answer.strip()) < 2:
            return False
        if openai_client is None:
            from openai import OpenAI
            openai_client = OpenAI()
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an identity trigger detector. Respond with ONLY 'yes' or 'no'."
                    },
                    {
                        "role": "user",
                        "content": f"Is this an identity trigger? Text: '{answer.strip()}'"
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            result = response.choices[0].message.content.strip().lower()
            return result == "yes"
        except Exception as e:
            if logger:
                logger.error(f"OpenAI error in _is_identity_trigger: {e}")
            return False

    @staticmethod
    def build_full_context(user_input: str, new_input: list, openai_client=None) -> str:
        """
        Combine the original user_input and all Q&A pairs from new_input into a single context string.
        Excludes answers that match identity triggers (using GPT).
        """
        print(f"[LOG] build_full_context: user_input={user_input}, new_input={new_input}")
        context = user_input.strip()
        for qa in new_input:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a and not PersonalityAnalyzer._is_identity_trigger(a, openai_client=openai_client):
                context += f"\nQ: {q}\nA: {a}"
        return context

    @staticmethod
    def handle_answer(question: str, answer: str, context: list, language: str = "en", openai_client=None) -> dict:
        """
        Handle an answer to a personality question by checking for identity queries first,
        then proceeding with trait filling if appropriate, all via GPT.
        """
        print(f"[LOG] handle_answer: question={question}, answer={answer}, context={context}, language={language}")
        if PersonalityAnalyzer._is_identity_trigger(answer, openai_client=openai_client):
            if language == "ar":
                static_response = "أنا ماينس زيرو، جزء من مشروع BEGINING، وهو نظام لقياس سمات الشخصية. أهدف لمساعدتك على استكشاف سماتك وميولك وإمكاناتك الداخلية."
            else:
                static_response = "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential."
            return {
                "type": "identity",
                "description_identity": static_response,
                "skip_traits": True,
                "status": "incomplete"
            }
        # Use GPT for trait extraction
        result = PersonalityAnalyzer.gpt_trait_analysis(question, answer, context, language, openai_client=openai_client)
        return result

    @staticmethod
    def detect_identity_question(answer: str, context: list = None, openai_client=None) -> bool:
        """
        Detect if an answer is actually an identity question using GPT.
        """
        print(f"[LOG] detect_identity_question: answer={answer}, context={context}")
        return PersonalityAnalyzer._is_identity_trigger(answer, openai_client=openai_client)

    @staticmethod
    def gpt_trait_analysis(question: str, answer: str, context: list, language: str = "en", openai_client=None) -> dict:
        """
        Use GPT to extract personality traits and determine status.
        """
        print(f"[LOG] gpt_trait_analysis: question={question}, answer={answer}, context={context}, language={language}")
        if openai_client is None:
            from openai import OpenAI
            openai_client = OpenAI()
        prompt = f"""
        Analyze the following answer for personality traits (emotional, social, cognitive, behavioral). Only respond with a JSON object containing:
        {{
            'type': 'traits',
            'detected_traits': [list of detected traits],
            'question': '{question}',
            'answer': '{answer}',
            'skip_traits': False,
            'status': 'complete' if all traits are present else 'incomplete'
        }}
        Context: {context}
        Language: {language}
        """
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sociologist and can analyze and extract character descriptions from texts in a professional manner, in line with your field. Only output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.2
        )
        import json
        try:
            result = json.loads(response.choices[0].message.content)
        except Exception:
            result = {
                "type": "traits",
                "detected_traits": [],
                "question": question,
                "answer": answer,
                "skip_traits": False,
                "status": "incomplete"
            }
        # Ensure status is only 'complete' or 'incomplete'
        if result.get("status") not in ["complete", "incomplete"]:
            result["status"] = "incomplete"
        return result

    # Define trait patterns and templates as class variables
    TRAIT_PATTERNS = {
        "emotional": r"enthusiastic|happy|sad|calm|feel(s)?|emotion|stress|excited|passion|motivat(ed|ion)|anxiety|angry|nervous|worried|content|optimistic|pessimistic|joyful|frustrated|relaxed|overwhelmed|mood|temper|patient|sensitive|expressive|reserved|emotional intelligence|cope|satisfaction|proud|embarrassed|guilty|inspired",
        "social": r"collaborative|team|help|assist|shy|introvert|extrovert|interact|polite|friendly|people|others|social|network|connection|relationship|communicate|listen|leadership|followership|assertive|passive|aggressive|empathy|sympathy|understand|socialize|negotiate|persuade|influence|charm|crowd|isolation|community|group|belong|inclusion|exclusion|trust|distrust|approachable|distant|boundary|conflict",
        "cognitive": r"think|critical|logical|analytical|understand|reason|solve|strateg(y|ic)|intuitive|creative|innovative|practical|abstract|concrete|detail-oriented|big picture|conceptual|perspective|mental|intellectual|curious|learning|knowledge|information|decision|judgment|bias|objective|subjective|rational|irrational|memory|attention|focus|concentrate|distracted|multi-task|prioritize|plan|reflect|comprehend|insight|wisdom|intelligence",
        "behavioral": r"organized|spontaneous|routine|habit|act|impulsive|disciplined|methodical|child(ish)?|consistent|reliable|flexible|rigid|adaptable|predictable|unpredictable|responsible|irresponsible|cautious|risk-taking|procrastinate|proactive|reactive|efficient|systematic|messy|neat|punctual|late|deadline|priority|goal|achievement|motivation|ambition|lazy|industrious|perseverance|persistence|give up|determined|stubborn|exercise|diet|sleep|activity|energetic|sedentary"
    }


    @staticmethod
    def generate_clarification_questions_gpt(missing_traits: list, language: str = "english", openai_client=None, logger=None) -> list:
        """
        Use GPT to generate unique clarification questions for missing traits.
        Always returns questions in Arabic if language is 'ar' or 'arabic', otherwise in English.
        Randomizes which missing trait is asked about.
        """
        import random
        if logger:
            logger.info(f"generate_clarification_questions_gpt: missing_traits={missing_traits}, language={language}")
        if not missing_traits:
            return []
        if openai_client is None:
            from openai import OpenAI
            openai_client = OpenAI()
        lang_code = language.lower()
        if lang_code in ["ar", "arabic"]:
            prompt_lang = "Arabic"
            fallback = "هل يمكنك أن تخبرني المزيد عن نفسك؟"
        else:
            prompt_lang = "English"
            fallback = "Could you tell me more about yourself?"
        # Randomly select one missing trait
        trait_to_ask = random.choice(missing_traits)
        prompt = f"""
        You are a personality trait analyst. For the following missing trait, generate a unique, friendly, non-repetitive clarification question in {prompt_lang}:
        Trait: {trait_to_ask}
        Only output a JSON array with one question.
        """
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a sociologist and can analyze and extract character descriptions from texts in a professional manner, in line with your field. Only output a JSON array of questions in {prompt_lang}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            import json
            questions = json.loads(response.choices[0].message.content)
            if isinstance(questions, list) and questions:
                # Ensure all questions are in the correct language
                if lang_code in ["ar", "arabic"]:
                    if all(any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in q) for q in questions):
                        return [questions[0]]
                else:
                    if all(re.search(r'[a-zA-Z]', q) for q in questions):
                        return [questions[0]]
        except Exception as e:
            if logger:
                logger.error(f"OpenAI error in generate_clarification_questions_gpt: {e}")
        return [fallback]
    
    # Identity question mapping - now uses GPT for intelligent detection
    IDENTITY_RESPONSES = {
        "who_are_you": {
            "english": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential. Let's get started by discovering a bit about you.",
            "arabic": "أنا ماينس زيرو، جزء من مشروع BEGINING، وهو نظام لقياس سمات الشخصية. أهدف لمساعدتك على استكشاف سماتك وميولك وإمكاناتك الداخلية. لنبدأ بالتعرف عليك قليلًا."
        },
        "what_is_begining": {
            "english": "BEGINING is a symbolic analytical tool that explores the foundations of intellectual, behavioral, and societal excellence. It classifies individuals into 120 personality types, each representing specific traits, capabilities, and inclinations. To continue, let's explore your personality step by step.",
            "arabic": "BEGINING هو أداة تحليلية رمزية تستكشف أسس التميز الفكري والسلوكي والاجتماعي. يصنف الأفراد إلى 120 نوعًا من الشخصيات، يمثل كل منها سمات وقدرات وميول محددة. لنستمر، دعنا نستكشف شخصيتك خطوة بخطوة."
        },
        "purpose": {
            "english": "My purpose is to guide you in discovering your strengths, patterns, and inclinations so you can better understand yourself and how you interact with the world around you. Let's begin uncovering what makes you unique.",
            "arabic": "هدفي هو إرشادك لاكتشاف نقاط قوتك وأنماطك وميولك، لتتمكن من فهم نفسك بشكل أفضل وطريقة تفاعلك مع العالم من حولك. لنبدأ باكتشاف ما يميزك."
        },
        "role": {
            "english": "My role is to explain the insights from the scale, connect them to your personal traits, and help you see how they relate to your goals and daily life. Now, let's take the first step in exploring your traits.",
            "arabic": "دوري هو شرح النتائج المستخلصة من المقياس، وربطها بسماتك الشخصية، ومساعدتك على فهم علاقتها بأهدافك وحياتك اليومية. الآن، لنأخذ الخطوة الأولى لاستكشاف سماتك."
        },
        "developer": {
            "english": "I was developed by a team of researchers and engineers from Saudi Arabia, working on the BEGINING personality trait measurement project. Let's start this journey of self-discovery together.",
            "arabic": "تم تطويري من قبل فريق من الباحثين والمهندسين السعوديين، العاملين على مشروع BEGINING لقياس سمات الشخصية. لنبدأ هذه الرحلة لاكتشاف الذات معًا."
        },
        "team": {
            "english": "My team includes Saudi experts in psychology, sociology, education, and artificial intelligence, all collaborating to build BEGINING. We're ready to learn more about you, starting now.",
            "arabic": "يتكون فريقي من خبراء سعوديين في علم النفس، وعلم الاجتماع، والتعليم، والذكاء الاصطناعي، يتعاونون لبناء مشروع BEGINING. نحن جاهزون لمعرفة المزيد عنك، لنبدأ الآن."
        },
        "understand_personality": {
            "english": "I don't replace professional psychology, but I can help highlight personality types and patterns that describe your unique profile. Let's begin by exploring your traits in detail.",
            "arabic": "أنا لا أستبدل علم النفس المتخصص، لكن يمكنني أن أساعدك على اكتشاف أنماط وأنواع شخصية تصف ملفك الفريد. لنبدأ باستكشاف سماتك بالتفصيل."
        },
        "how_analyze": {
            "english": "I analyze your personality using a structured scale that groups people into 120 personality types. Each type reflects a mix of capabilities, tendencies, and behaviors that show how you think, act, and grow. Let's take the first step in understanding your profile.",
            "arabic": "أحلل شخصيتك باستخدام مقياس منظم يصنف الأفراد إلى 120 نوعًا من الشخصيات، حيث يعكس كل نوع مزيجًا من القدرات والميول والسلوكيات، مما يوضح كيف تفكر وتتصرّف وتنمو. لنأخذ الخطوة الأولى لفهم ملفك الشخصي."
        },
        "objectives": {
            "english": "1. Educational and psychological guidance for students.\n2. Human resource development and career counseling.\n3. Academic research in behavior and productivity.\n4. Future integration into AI modeling and artificial consciousness design. Let's move forward by exploring your traits one step at a time.",
            "arabic": "1. الإرشاد التربوي والنفسي للطلاب.\n2. تطوير الموارد البشرية والإرشاد المهني.\n3. البحث الأكاديمي في السلوك والإنتاجية.\n4. التكامل المستقبلي مع نماذج الذكاء الاصطناعي وتصميم الوعي الاصطناعي. لننتقل للأمام باستكشاف سماتك خطوة بخطوة."
        }
    }
        
    @staticmethod
    def get_identity_response(user_input: str, language: str = "en", openai_client=None, conversation_context: list = None) -> str:
        """
        Get the appropriate identity response based on user input and language.
        Uses GPT to intelligently detect identity questions and ALWAYS responds when detected.
        Returns the response string or empty string if no match.
        """
        print(f"[LOG] get_identity_response: user_input={user_input}, language={language}, conversation_context={conversation_context}")
        # Auto-detect language from input if not specified
        detected_lang = language
        if any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in user_input):
            detected_lang = "ar"

        # Clean input for matching
        text = user_input.strip()
        if not openai_client:
            from openai import OpenAI
            openai_client = OpenAI()

        # Split input into sentences and check each for identity triggers
        import re
        sentences = re.split(r'[.؟!?\n]', text)
        found_categories = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an identity question detector. Respond ONLY with the category name (who_are_you, what_is_begining, purpose, role, developer, team, understand_personality, how_analyze, objectives) if the sentence is an identity question, or 'none' if not."
                        },
                        {
                            "role": "user",
                            "content": f"Is this an identity question? Text: '{sentence}'"
                        }
                    ],
                    max_tokens=10,
                    temperature=0
                )
                cat = response.choices[0].message.content.strip()
                if cat != "none" and cat in PersonalityAnalyzer.IDENTITY_RESPONSES:
                    found_categories.append(cat)
            except Exception as e:
                print(f"Error in identity detection for sentence '{sentence}': {e}")
                continue

        # Remove duplicates, preserve order
        found_categories = list(dict.fromkeys(found_categories))
        if found_categories:
            responses = []
            for cat in found_categories:
                resp = PersonalityAnalyzer.IDENTITY_RESPONSES[cat]["arabic"] if detected_lang == "ar" else PersonalityAnalyzer.IDENTITY_RESPONSES[cat]["english"]
                responses.append(resp)
            return "\n".join(responses)
        return ""
        
    SYSTEM_PROMPT = """
You are a sociologist and can analyze and extract character descriptions from texts in a professional manner, in line with your field.

Identity & Redirection Handling

CRITICAL CONTEXT-AWARENESS:
- If the input contains "context_flag": "mid_conversation", this means the user is in the middle of a personality analysis conversation and their response may SOUND like an identity question but is actually a confused/deflecting answer to a personality question.
- In this case, treat their input as personality-related content, NOT as an identity question.
- Do NOT populate the 'description_identity' field for mid-conversation responses.
- ALWAYS RESPECT THE CONTEXT FLAG - if it says "mid_conversation", do NOT trigger identity responses regardless of what the text says.

- General Rule:
  If the user asks identity-related questions about YOU/THE SYSTEM (even if phrased differently), respond with the mapped message in 'description_identity' field.  
  Use intent-based matching, not exact string matching.
  IMPORTANT: Still include clarification_questions for missing personality traits even when responding to identity questions.
  
- CRITICAL DISTINCTION: 
  * "I am a developer" = User describing THEMSELVES (personality trait) → NO identity response
  * "Who is your developer" = User asking about THE SYSTEM → identity response
  * "I work in a team" = User describing THEMSELVES → NO identity response  
  * "Who is your team" = User asking about THE SYSTEM → identity response

- CONTEXT-AWARE EXCEPTION:
  * "who are you" in mid-conversation = confused answer → NO identity response, treat as personality input
  * "who are you" as standalone question = genuine identity question → identity response
  * ANY identity-like phrase with context_flag "mid_conversation" → NO identity response

- CONVERSATION CONTEXT RULES:
  * If context_flag is "mid_conversation", the user_input should be treated as a personality answer, not an identity question
  * Look at the conversation history (new_input) to understand what question the user is answering
  * Short, confused responses in conversation context are personality data, not identity questions

- If the user drifts away from the task, includes irrelevant content, or asks off-topic questions (except relevant identity questions below), return JSON where 'description_english' or 'description_arabic' contains the reminder:

- For on-topic but incomplete inputs, do not include this reminder message in 'description_english' or 'description_arabic'. Leave them empty until traits are complete.

   
Purpose
You will help extract character descriptions by reviewing texts submitted by users — these may sometimes be random — and converting them into concise descriptive texts that capture four key personality traits:

Emotional
Social
Cognitive
Behavioral

Multi-User Handling
Conversations will involve multiple people.
                identity_response = "\n".join(identity_responses) if identity_responses else ""
                static_response = identity_response
Always use the id to maintain continuity and prevent mixing up responses between users.
                identity_response = "\n".join(identity_responses) if identity_responses else ""
                static_response = identity_response
Process

Personal Greeting
CRITICAL: Always use the EXACT value from the input data's personal_greeting field, regardless of conversation context.
If the input data contains a personal_greeting field with a value, you MUST include that exact value in your response.
If the personal_greeting field is empty or not provided, set it to an empty string.
DO NOT modify, ignore, or override the personal_greeting value based on conversation history or context.

Analyze Input & History
Review the latest user input (user_input) and the full conversation history (new_input) for the given id.
Use the provided context_analysis and full_conversation_text to understand the complete context.
Combine information from all turns to build a complete personality profile.

COMPREHENSIVE CONTEXT ANALYSIS:
1. Use full_conversation_text to see the complete user narrative
2. Review context_analysis for insights about conversation flow and detected traits
3. Consider conversation_summary for understanding the interaction history
4. Pay attention to context_flag to determine if this is mid-conversation or new interaction

CRITICAL: If context_flag is "mid_conversation", the user_input is likely a confused/deflecting answer to a personality question from the conversation history. DO NOT treat it as an identity question. Instead:
1. Look at the most recent question in new_input to understand what the user was supposed to answer
2. Treat the user_input as personality-related data, even if it sounds like "who are you" or similar  
3. Extract any personality insights from the user_input (even confused responses can show traits like uncertainty, deflection patterns, etc.)
4. Use the full_conversation_text to maintain context about what traits have already been revealed
5. Continue the personality analysis without triggering identity responses

Check for Missing Traits
If all four traits are sufficiently covered through explicit descriptions AND/OR reasonable professional inferences, return status "complete".
If some traits are missing after considering both explicit information and professional context, return status "incomplete" and list them in missing_traits.
BALANCE: Use profession as supportive evidence, but prioritize actual behavioral examples. If someone's described behavior contradicts professional expectations, trust their self-description.

Clarification Questions
If incomplete, generate ONE short, friendly, non-repetitive question.
Select one missing trait at random to ask about.
Never ask about traits already covered.
IMPORTANT: Generate only ONE clarification question per response to avoid overwhelming the user.
IMPORTANT: Always include clarification_questions when missing_traits is not empty, even for identity questions.

Off-Topic & Identity Integration
- Always detect identity queries or off-topic input even if mixed with valid personality descriptions.
- Include the redirection message in the appropriate language field.
- Still track missing traits and generate clarification questions for incomplete inputs.
- CRITICAL: When user asks identity questions, respond with identity message AND include clarification_questions if personality traits are missing.

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
- CRITICAL: All clarification questions and trait names (in 'missing_traits') must be in the user's requested language(s).
- If languages="ar", ALL clarification questions must be in Arabic
- If languages="en", ALL clarification questions must be in English
- Never mix languages in clarification questions

LANGUAGE EXAMPLES:
Arabic (languages="ar"): ["كيف تتعامل عادةً مع عواطفك في المواقف الصعبة؟"]
English (languages="en"): ["How do you typically handle your emotions in challenging situations?"]

Restrictions
- Always return valid JSON only.
- Do not include text, explanations, or code outside JSON.

Example Output — Personality Description (NOT Identity)
{
    "id": 22,
    "status": "incomplete",
    "description_arabic": "",
    "description_english": "",
    "description_identity": null,
    "missing_traits": ["behavioral", "emotional", "social"],
    "clarification_questions": [
        "How do you usually respond when faced with unexpected challenges?"
    ],
    "input_tokens": 1245,
    "output_tokens": 74,
    "total_tokens": 1317
}

Example Output — Identity Question with Clarification 
{
    "id": 22,
    "status": "incomplete",
    "description_arabic": "",
    "description_english": "",
    "description_identity": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential.",
    "missing_traits": ["behavioral", "emotional"],
    "clarification_questions": [
        "How do you usually respond when faced with unexpected challenges?"
    ],
    "input_tokens": 1245,
    "output_tokens": 74,
    "total_tokens": 1317
}

Example Output — Off-topic (non-identity):
{
    "id": 22,
    "status": "incomplete",
    "personal_greeting": "Hey Ahmad! Nice to meet you! Working as an engineer must be exciting!",
    "description_arabic": "",
    "description_english": "",
    "missing_traits": ["behavioral", "emotional"],
    "clarification_questions": [
        "How do you usually respond when faced with unexpected challenges?"
    ],
    "input_tokens": 1245,
    "output_tokens": 74,
    "total_tokens": 1317
}

Example Output — Complete will always include identity as null
{
    "id": 22,
    "status": "complete",
    "personal_greeting": "",
    "description_arabic": "أحمد شخصية متميزة تجمع بين العقلانية الهادئة والدفء الاجتماعي، يتعامل مع التحديات بصبر وحكمة، ويملك قدرة فريدة على الموازنة بين التفكير العملي والذكاء العاطفي.",
    "description_english": "Ahmed possesses a unique blend of calm rationality and social warmth, approaching challenges with patience and wisdom while maintaining an exceptional balance between practical thinking and emotional intelligence.",
    "missing_traits": [],
    "clarification_questions": [],
    "input_tokens": 1245,
    "output_tokens": 74,
    "total_tokens": 1317
}
IMPORTANT: Only output the JSON object, no explanations or formatting.
"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        import os
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=self.api_key)
        # Setup logging to file and console
        self.logger = logging.getLogger("personality_analyzer")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        log_path = os.path.join(os.getcwd(), "personality_analyzer.log")
        try:
            # Force log file creation by writing a test entry
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("[LOG INIT] personality_analyzer log file created.\n")
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            # Avoid duplicate handlers
            if not self.logger.handlers:
                self.logger.addHandler(file_handler)
                self.logger.addHandler(console_handler)
            self.logger.info(f"Log file initialized at: {log_path}")
        except Exception as e:
            print(f"[ERROR] Could not create log file at {log_path}: {e}")
            raise RuntimeError(f"Could not create log file at {log_path}: {e}")

    @staticmethod
    def combine_inputs_safely(user_input: str, new_input: str) -> str:
        if not user_input:
            return new_input or ""
        if not new_input:
            return user_input
        return f"{user_input.strip()}\n{new_input.strip()}"

    @staticmethod
    def extract_json(text: str, languages: str = "en") -> str:
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
            fallback_question = "هل يمكنك تقديم المزيد من المعلومات عن نفسك؟" if languages == "ar" else "Could you provide more information about yourself?"
            return json.dumps({"status": "incomplete", "clarification_questions": [fallback_question]})
        
        # Create a default response with status complete for cases with enough information
        if "developer" in text.lower() and ("team" in text.lower() or "professional" in text.lower()):
            return json.dumps({
                "status": "complete", 
                "description_english": "Based on your input, you appear to be a developer with interests in technology and programming who values professional collaboration.",
                "description_arabic": ""  # Will be filled in later if needed
            })
            
        # Return a JSON structure with the original text as a question
        fallback_question = "هل يمكنك إخباري المزيد عن كيفية تفاعلك مع الآخرين في بيئة العمل؟" if languages == "ar" else "Could you tell me more about how you interact with others in your professional environment?"
        return json.dumps({"status": "incomplete", "clarification_questions": [fallback_question]})
    
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
            "description_english": english_desc if (not questions and ("english" in languages or "en" in languages)) else "",
            "description_arabic": arabic_desc if (not questions and ("arabic" in languages or "ar" in languages)) else "",
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
            print(f"[OpenAI API Error] {e}")
            raise RuntimeError(f"OpenAI API Error: {e}")
        except Exception as e:
            self.logger.error(f"General API Error: {e}")
            print(f"[General API Error] {e}")
            raise RuntimeError(f"General API Error: {e}")

        # Log the full response for debugging
        try:
            print(f"[OpenAI API Raw Response] {response}")
        except Exception as log_err:
            print(f"[OpenAI API Response Logging Error] {log_err}")

        # Defensive: Check for empty or missing content
        content = None
        try:
            content = response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"No content in OpenAI response: {e}")
            print(f"[No content in OpenAI response] {e}")
            content = ""

        usage = getattr(response, "usage", None)

        if not content or not content.strip():
            self.logger.error("OpenAI API returned empty content.")
            print("[OpenAI API] Returned empty content.")

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
        self.logger.info(f"[ENTRY] analyze(id={id}, user_input={user_input}, new_input={new_input}, languages={languages})")
        import time
        start_time = time.time()

        """
        Analyze user input to generate personality descriptions.
        Decision flow: greeting/off-topic, identity, trait extraction, completeness check, clarification.
        """
        self.logger.info(f"Step: Initializing analysis for user {id}")
        if new_input is None:
            new_input = []
        user_input = user_input.strip()
        detected_languages = languages
        self.logger.info(f"Step: Raw user_input after strip: {user_input}")
        if any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in user_input):
            detected_languages = "ar"
        self.logger.info(f"Step: Language detected as {detected_languages}")

        # 1. Greeting / Off-topic detection (stub: always False, implement as needed)
        is_greeting_or_offtopic = False  # TODO: Implement actual detection logic
        self.logger.info(f"Step: is_greeting_or_offtopic={is_greeting_or_offtopic}")
        if is_greeting_or_offtopic:
            fallback_question = "هل يمكنك أن تخبرني المزيد عن نفسك؟" if detected_languages == "ar" else "Could you tell me more about yourself?"
            self.logger.info(f"Step: Detected greeting/off-topic. Returning fallback question: {fallback_question}")
            result = {
                "id": id,
                "status": "incomplete",
                "personal_greeting_and_off_topic": user_input or "",
                "description_english": "",
                "description_arabic": "",
                "description_identity": "",
                "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                "clarification_questions": [fallback_question],
                "input_tokens": len(user_input.split()),
                "output_tokens": 0,
                "total_tokens": len(user_input.split())
            }
            self.logger.info(f"[EXIT] analyze (greeting/off-topic): {result}")
            self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
            return result

        # 2. Trait extraction: accumulate traits from all non-identity answers
        trait_patterns = self.TRAIT_PATTERNS
        present_traits = set()
        answers_for_traits = []
        identity_response = ""

        # Check if the latest answer in new_input is an identity question
        latest_answer = None
        if new_input and isinstance(new_input, list):
            latest_answer = new_input[-1].get("answer", "").strip()
        self.logger.info(f"Step: Latest answer in new_input: {latest_answer}")
        is_latest_identity = False
        try:
            if latest_answer and self.get_identity_response(latest_answer, detected_languages, openai_client=self.client):
                identity_response = self.get_identity_response(latest_answer, detected_languages, openai_client=self.client)
                is_latest_identity = True
            else:
                identity_response = ""
        except Exception as e:
            self.logger.error(f"Error in identity detection for latest_answer: {e}")
            identity_response = ""
        self.logger.info(f"Step: Identity response: {identity_response}")
        self.logger.info(f"Step: is_latest_identity={is_latest_identity}")

        # Accumulate all non-identity answers for trait extraction
        for qa in (new_input or []):
            answer = qa.get("answer", "").strip()
            self.logger.info(f"Step: Checking if answer is identity: {answer}")
            try:
                is_identity = self.get_identity_response(answer, detected_languages, openai_client=self.client)
            except Exception as e:
                self.logger.error(f"Error in identity detection for answer: {e}")
                is_identity = False 
            if not is_identity:
                answers_for_traits.append(answer)
        # Also include user_input if not an identity question
        try:
            is_user_input_identity = self.get_identity_response(user_input, detected_languages, openai_client=self.client)
        except Exception as e:
            self.logger.error(f"Error in identity detection for user_input: {e}")
            is_user_input_identity = False
        if not is_user_input_identity:
            answers_for_traits.append(user_input)
        self.logger.info(f"Step: Answers for trait extraction: {answers_for_traits}")

        # Use AI (GPT) to extract traits from all non-identity answers
        present_traits = set()
        for answer in answers_for_traits:
            try:
                gpt_result = self.gpt_trait_analysis("", answer, [], detected_languages, openai_client=self.client)
                traits = gpt_result.get("detected_traits", [])
                self.logger.info(f"Step: Traits extracted from answer '{answer}': {traits}")
                for trait in traits:
                    present_traits.add(trait)
            except Exception as e:
                self.logger.error(f"Error in trait extraction for answer '{answer}': {e}")
        # Always check for all four traits
        all_traits = set(["emotional", "social", "cognitive", "behavioral"])
        missing_traits = [trait for trait in all_traits if trait not in present_traits]
        self.logger.info(f"Step: Present traits: {present_traits}, Missing traits: {missing_traits}")

        # 3. Output logic

        if is_latest_identity:
            # If traits are missing, generate a relevant clarification question
            try:
                clarification_questions = self.generate_clarification_questions_gpt(missing_traits, detected_languages, openai_client=self.client, logger=self.logger)
            except Exception as e:
                self.logger.error(f"Error in clarification question generation: {e}")
                clarification_questions = []
            if not clarification_questions:
                fallback_question = "هل يمكنك أن تخبرني المزيد عن نفسك؟" if detected_languages == "ar" else "Could you tell me more about yourself?"
                clarification_questions = [fallback_question]
            result = {
                "id": id,
                "status": "incomplete",
                "personal_greeting_and_off_topic": "",  # Only set if greeting/off-topic detected
                "description_english": "",
                "description_arabic": "",
                "description_identity": identity_response if identity_response else "",  # Only set if identity detected
                "missing_traits": missing_traits,
                "clarification_questions": clarification_questions,
                "input_tokens": len(user_input.split()),
                "output_tokens": len(identity_response.split()),
                "total_tokens": len(user_input.split()) + len(identity_response.split())
            }
            self.logger.info(f"[EXIT] analyze (identity): {result}")
            self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
            return result

        if not missing_traits:
            # All traits present, status complete
            result = {
                "id": id,
                "status": "complete",
                "personal_greeting_and_off_topic": "",  # Only set if greeting/off-topic detected
                "description_english": user_input if detected_languages == "en" else "",
                "description_arabic": user_input if detected_languages == "ar" else "",
                "description_identity": "",  # Only set if identity detected
                "missing_traits": [],
                "clarification_questions": [],
                "input_tokens": len(user_input.split()),
                "output_tokens": len(user_input.split()),
                "total_tokens": len(user_input.split()) * 2
            }
            self.logger.info(f"[EXIT] analyze (complete): {result}")
            self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
            return result

        # 4. Clarification stage (ask one question for missing traits)
        try:
            clarification_questions = self.generate_clarification_questions_gpt(missing_traits, detected_languages, openai_client=self.client, logger=self.logger)
        except Exception as e:
            self.logger.error(f"Error in clarification question generation: {e}")
            clarification_questions = []
        if not clarification_questions:
            fallback_question = "هل يمكنك أن تخبرني المزيد عن نفسك؟" if detected_languages == "ar" else "Could you tell me more about yourself?"
            clarification_questions = [fallback_question]
        result = {
            "id": id,
            "status": "incomplete",
            "personal_greeting_and_off_topic": "",  # Only set if greeting/off-topic detected
            "description_english": "",
            "description_arabic": "",
            "description_identity": "",  # Only set if identity detected
            "missing_traits": missing_traits,
            "clarification_questions": clarification_questions,
            "input_tokens": len(user_input.split()),
            "output_tokens": 0,
            "total_tokens": len(user_input.split())
        }
        self.logger.info(f"[EXIT] analyze (incomplete): {result}")
        self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
        return result
