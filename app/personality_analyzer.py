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
    def remove_questions(text: str) -> str:
        """
        Removes any sentences containing question marks (Arabic or English) or question words from the text.
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!؟?\n]', text)
        question_words = [
            'هل', 'فهل', 'لماذا', 'كيف', 'متى', 'أين', 'ما', 'ماذا', 'أي', 'هل ترغب', 'هل تريد',
            'do you', 'would you', 'can you', 'could you', 'will you', 'are you', 'is it', 'shall we', 'should you', 'why', 'how', 'when', 'where', 'what', 'which'
        ]
        filtered = []
        for s in sentences:
            s_strip = s.strip()
            if not s_strip:
                continue
            if '?' in s_strip or '؟' in s_strip:
                continue
            if any(qw in s_strip for qw in question_words):
                continue
            filtered.append(s_strip)
        return '. '.join(filtered).strip()
        
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
        Analyze the following answer for personality traits and categorize them into exactly these four categories:
        - emotional: emotional intelligence, mood, feelings, motivation, stress handling
        - social: teamwork, leadership, communication, interpersonal skills
        - cognitive: analytical thinking, problem-solving, learning, decision-making
        - behavioral: work habits, organization, consistency, actions, routines
        
        Only respond with a JSON object containing:
        {{
            'type': 'traits',
            'detected_traits': [list containing only: "emotional", "social", "cognitive", "behavioral"],
            'question': '{question}',
            'answer': '{answer}',
            'skip_traits': false,
            'status': 'incomplete'
        }}
        
        Answer to analyze: {answer}
        Context: {context}
        Language: {language}
        """
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sociologist and can analyze and extract character descriptions from texts in a professional manner, in line with your field. Only output valid JSON with trait categories: emotional, social, cognitive, behavioral."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )
            import json
            content = response.choices[0].message.content
            print(f"[LOG] GPT trait analysis response: {content}")
            
            # Handle markdown code blocks
            if content.startswith("```"):
                # Extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    # Try to find JSON without code blocks
                    json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
            
            result = json.loads(content.strip())
            
            # Ensure detected_traits only contains the four main categories
            valid_traits = ["emotional", "social", "cognitive", "behavioral"]
            detected_traits = result.get("detected_traits", [])
            filtered_traits = [trait for trait in detected_traits if trait in valid_traits]
            result["detected_traits"] = filtered_traits
            
        except Exception as e:
            print(f"[LOG] Error in gpt_trait_analysis: {e}")
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
            content = response.choices[0].message.content
            if logger:
                logger.info(f"OpenAI response content: {content}")
            
            # Handle empty or null content
            if not content or content.strip() == "":
                if logger:
                    logger.warning("OpenAI returned empty content")
                return [fallback]
            
            questions = json.loads(content.strip())
            if isinstance(questions, list) and questions:
                # Ensure all questions are in the correct language
                if lang_code in ["ar", "arabic"]:
                    if all(any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in q) for q in questions):
                        return [questions[0]]
                else:
                    if all(re.search(r'[a-zA-Z]', q) for q in questions):
                        return [questions[0]]
        except json.JSONDecodeError as e:
            if logger:
                logger.error(f"JSON decode error in generate_clarification_questions_gpt: {e}, content: {content if 'content' in locals() else 'No content'}")
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
        
    def _extract_personality_content_from_mixed_input(self, text: str, language: str = "en") -> str:
        """
        AI-powered extraction of personality-related content from mixed input.
        Intelligently separates personality descriptions from identity questions and other content.
        """
        if not text.strip():
            return ""
        
        # Use GPT for intelligent content separation
        if not hasattr(self, 'client') or not self.client:
            from openai import OpenAI
            self.client = OpenAI()

        system_prompt = """You are an expert content analyzer for a personality analysis system. Your job is to extract ONLY personality-related content from mixed input.

EXTRACT THESE PERSONALITY ELEMENTS:
- Personal traits: "I am introverted", "I'm very social", "أنا شخص هادئ"
- Behaviors: "I like to help others", "I prefer working alone", "أحب العمل الجماعي"
- Social preferences: "I like going out with friends", "I enjoy social gatherings", "أحب اللقاءات الاجتماعية"
- Activity preferences that reveal personality: "I enjoy reading", "I dislike crowds", "أفضل الأنشطة الهادئة"
- Emotional patterns: "I get stressed easily", "I'm usually optimistic", "أشعر بالقلق أحيانًا"
- Social tendencies: "I make friends easily", "I'm shy around new people", "أتفاعل بسهولة"
- Work styles: "I'm detail-oriented", "I like big picture thinking", "أركز على التفاصيل"
- Decision-making: "I think things through", "I go with my gut", "أتخذ قرارات سريعة"
- Lifestyle choices that show personality: "I prefer quiet evenings", "I love parties", "أحب السهر مع الأصدقاء"

IGNORE THESE NON-PERSONALITY ELEMENTS:
- Identity questions: "who are you", "what is BEGINING", "من انت"
- Greetings: "hello", "hi", "مرحبا"
- Pure off-topic: weather facts, news, general conversation unrelated to personal preferences
- Technical questions: how the system works (unless describing personal work style)

RESPONSE FORMAT:
- If personality content found: Return ONLY the personality-related parts
- If no personality content: Return "EMPTY"
- Preserve the original language and phrasing
- Clean up grammar but keep the meaning intact

EXAMPLES:
Input: "Hello, I am an introverted software developer who likes working alone"
Output: "I am an introverted software developer who likes working alone"

Input: "I like go out with friends"
Output: "I like go out with friends"

Input: "مرحبا، من انت؟ أنا شخص اجتماعي وأحب العمل مع الفريق"
Output: "أنا شخص اجتماعي وأحب العمل مع الفريق"

Input: "Who are you and what is BEGINING?"
Output: "EMPTY"

Input: "What color is the sky?"
Output: "EMPTY"

Input: "I was reading about AI, by the way I'm very analytical and detail-oriented in my work"
Output: "I'm very analytical and detail-oriented in my work"

IMPORTANT: Social activities and preferences (like "going out with friends") are personality-related. Be generous in extracting personality content but strict about ignoring pure identity questions, greetings, and factual off-topic questions."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract personality content from: '{text}'"}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            if result.upper() == "EMPTY" or not result:
                return ""
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in AI content extraction: {e}")
            # Fallback: return original text if AI fails
            return text
    
    def _pattern_based_content_separation(self, text: str) -> str:
        """
        Fallback method for content separation using patterns.
        """
        # Split by common sentence separators
        import re
        sentences = re.split(r'[.؟!?\n]', text)
        personality_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Enhanced identity question patterns
            identity_patterns = [
                # Arabic patterns
                r'من\s+انت|اخبرني\s+عن\s+نفسك|عرف\s+نفسك|اشرح\s+لي\s+من\s+انت',
                r'ما\s+هو\s+مشروع|اشرح\s+مشروع|عن\s+المشروع|ما\s+هو\s+بيجينينج',
                r'ما\s+هدفك|ما\s+هو\s+هدفك|لماذا\s+موجود|ما\s+دورك|ايش\s+دورك',
                r'من\s+طورك|من\s+صنعك|من\s+فريقك|كيف\s+تعمل|كيف\s+تحلل',
                
                # English patterns  
                r'who\s+are\s+you|what\s+are\s+you|tell\s+me\s+about\s+yourself|introduce\s+yourself',
                r'what\s+is\s+begining|explain\s+begining|about\s+this\s+project|what\s+is\s+this\s+system',
                r'what\s+is\s+your\s+purpose|why\s+do\s+you\s+exist|what\s+do\s+you\s+do|what\s+is\s+your\s+role',
                r'who\s+made\s+you|who\s+created\s+you|who\s+developed\s+you|who\s+is\s+your\s+team',
                r'how\s+do\s+you\s+work|how\s+do\s+you\s+analyze|what\s+is\s+your\s+method',
            ]
            
            is_identity = False
            for pattern in identity_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    is_identity = True
                    break
            
            if not is_identity:
                personality_sentences.append(sentence)
        
        # Rejoin the personality sentences
        result = '. '.join(personality_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result.strip()

    def _is_greeting(self, text: str, language: str = "en") -> bool:
        """
        Helper function to determine if text is a greeting.
        """
        if not text.strip():
            return False
            
        text_lower = text.lower().strip()
        
        # Arabic greeting patterns
        arabic_greetings = [
            "مرحبا", "اهلا", "السلام عليكم", "صباح الخير", "مساء الخير",
            "هلا", "اهلين", "حياك", "كيف حالك", "كيفك", "شلونك",
            "يسعد صباحك", "يسعد مساءك", "تسلم", "نهارك سعيد"
        ]
        
        # English greeting patterns  
        english_greetings = [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "how do you do", "nice to meet you", "pleased to meet you",
            "greetings", "howdy", "what's up", "how's it going"
        ]
        
        # Check for greetings
        if language.lower() in ["ar", "arabic"]:
            for greeting in arabic_greetings:
                if greeting in text_lower:
                    return True
        else:
            for greeting in english_greetings:
                if greeting in text_lower:
                    return True
                    
        return False

    @staticmethod
    def get_identity_response(user_input: str, language: str = "en", openai_client=None, conversation_context: list = None) -> str:
        """
        Get the appropriate identity response based on user input and language.
        Uses GPT to intelligently detect and categorize identity questions in multiple languages.
        Can handle multiple identity questions and create friendly combined responses.
        Returns the response string or empty string if no match.
        """
        print(f"[LOG] get_identity_response: user_input={user_input}, language={language}, conversation_context={conversation_context}")
        
        # Auto-detect language from input if not specified
        detected_lang = language
        if any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in user_input):
            detected_lang = "ar"

        # Clean input for matching
        text = user_input.strip()
        if not text:
            return ""
            
        if not openai_client:
            from openai import OpenAI
            openai_client = OpenAI()

        # Enhanced prompt for detecting identity questions in ANY context (stories, articles, etc.)
        system_prompt = """You are a PRECISE identity question detector for a personality analysis system called "BEGINING" with an AI assistant named "Minus Zero".

MISSION: Detect ONLY genuine identity questions about the SYSTEM/AI. Do NOT misclassify personality descriptions as identity questions.

CATEGORIES:
- who_are_you: Questions about the AI's identity (who are you, what are you, tell me about yourself, introduce yourself)
- what_is_begining: Questions about the BEGINING project/system (what is BEGINING, explain BEGINING, about this project)
- purpose: Questions about the AI's purpose/mission (what's your purpose, why do you exist, what do you do)
- role: Questions about the AI's role/function (what's your role, how do you help, what's your job)
- developer: Questions about who created/developed the system (who made you, who created you, who built you)
- team: Questions about the development team (who's your team, tell me about your creators)
- understand_personality: Questions about how personality analysis works (how do you understand personality, can you replace psychology)
- how_analyze: Questions about the analysis methodology (how do you analyze, what's your method, how does this work)
- objectives: Questions about goals/applications (what are your objectives, what's this for, applications)

CRITICAL: ONLY detect questions ABOUT THE SYSTEM/AI, NOT personality descriptions.

EXAMPLES OF IDENTITY QUESTIONS (detect these):

SMART DETECTION RULES:
1. Look for identity questions ANYWHERE in the text, even if mixed with other content
2. Detect questions hidden in stories: "I was wondering who you are while reading this article..."
3. Catch indirect questions: "Could you explain a bit about yourself before we start?"
4. Find embedded questions: "In my research about AI systems, I want to know what BEGINING is..."
5. Recognize conversational patterns: "Before we continue, tell me about your purpose..."
6. Handle multiple languages flexibly (Arabic, English, mixed)
7. Be context-aware: even if 90% is other content, find the 10% identity question

RETURN FORMAT:
- Return ALL applicable categories separated by commas if identity questions found
- Return "none" ONLY if there are absolutely NO identity questions about the system
- Be generous in detection - better to catch false positives than miss real questions

Examples:
"من انت" → who_are_you
"Who are you and what is your purpose?" → who_are_you,purpose
"How do you analyze personalities?" → how_analyze
"What is BEGINING?" → what_is_begining
"Who created you?" → developer
"Tell me about your methodology" → how_analyze
"I want to know how you work" → how_analyze
"I'm curious about your method" → how_analyze
"Can you explain how you analyze?" → how_analyze
"I'm a scientist but I want to know how you work" → how_analyze
"I love data, but tell me about your approach" → how_analyze

EXAMPLES OF PERSONALITY DESCRIPTIONS (DO NOT detect these):
"I enjoy working with data and solving problems" → none
"I love analytical work and finding patterns" → none  
"I'm someone who likes to analyze things" → none
"I find satisfaction in solving complex problems" → none
"I work with data and enjoy finding insights" → none
"I'm analytical and detail-oriented" → none
"I like to understand how things work" → none
"I'm a data scientist who loves patterns" → none

STRICT RULES:
1. Look for questions that are ABOUT the system/AI, not descriptions of the user's personality
2. If someone describes their own analytical nature, that's personality data, NOT an identity question
3. Only detect actual questions or requests for information about the system
4. User describing their work/interests/traits = personality data (return "none")
5. User asking about system's work/purpose/identity = identity question (return category)
6. Mixed content: "I'm X, but I want to know how you work" = identity question (detect the question part)
7. Watch for phrases like "I want to know", "I'm curious about", "tell me about", "but I want to know" referring to the system
8. Key detection phrases for mixed content: "want to know how you", "curious about your", "tell me about your", "explain your"

RETURN FORMAT:
- Return applicable categories separated by commas if genuine identity questions found
- Return "none" if NO identity questions about the system (even if analytical language is present)
- Be PRECISE, not generous - avoid false positives that confuse personality data with identity questions"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text: '{text}'"}
                ],
                max_tokens=50,
                temperature=0
            )
            
            categories = response.choices[0].message.content.strip().lower()
            print(f"[LOG] GPT identified categories: {categories}")
            
            if categories == "none" or not categories:
                print(f"[LOG] No identity question detected")
                return ""
            
            # Split multiple categories
            category_list = [cat.strip() for cat in categories.split(',') if cat.strip()]
            
            # Generate combined friendly response
            response_text = PersonalityAnalyzer._create_combined_identity_response(category_list, detected_lang)
            
            if response_text:
                print(f"[LOG] Returning combined identity response for categories: {category_list}")
                return response_text
            else:
                print(f"[LOG] No valid categories found in: {categories}")
                return ""
                
        except Exception as e:
            print(f"[LOG] Error in GPT identity detection: {e}")
            # Fallback to pattern matching for critical identity questions
            import re
            detected_categories = []
            
            fallback_patterns = [
                (r'من\s+انت|اخبرني\s+عن\s+نفسك|عرف\s+نفسك', 'who_are_you'),
                (r'who\s+are\s+you|what\s+are\s+you|tell\s+me\s+about\s+yourself|introduce\s+yourself', 'who_are_you'),
                (r'ما\s+هو\s+مشروع|اشرح\s+مشروع|عن\s+المشروع|ما\s+هو\s+بيجينينج', 'what_is_begining'),
                (r'what\s+is\s+begining|explain\s+begining|about\s+this\s+project', 'what_is_begining'),
                (r'ما\s+هدفك|ما\s+هو\s+هدفك|لماذا\s+موجود', 'purpose'),
                (r'what\s+is\s+your\s+purpose|why\s+do\s+you\s+exist|what\s+do\s+you\s+do', 'purpose'),
                (r'كيف\s+تعمل|كيف\s+تحلل', 'how_analyze'),
                (r'how\s+do\s+you\s+work|how\s+do\s+you\s+analyze', 'how_analyze'),
            ]
            
            for pattern, category in fallback_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if category not in detected_categories:
                        detected_categories.append(category)
            
            if detected_categories:
                response_text = PersonalityAnalyzer._create_combined_identity_response(detected_categories, detected_lang)
                print(f"[LOG] Fallback pattern match for categories: {detected_categories}")
                return response_text
            
            return ""
    
    @staticmethod
    def get_greeting_or_offtopic_response(user_input: str, language: str = "en", openai_client=None) -> str:
        """
        Handles greetings like off-topic: returns a formal, complete, positive statement (no questions), similar to off-topic style.
        Enforces that no questions are present in the response.
        """
        if not user_input.strip():
            return ""

        # Templates for greeting responses (no questions, only positive statements)
        english_templates = [
            "It's wonderful to meet you! I'm excited to help you discover your unique personality traits and what makes you special.",
            "Hello! I'm here to help you explore your unique personality traits and strengths.",
            "Welcome! I'm delighted to assist you in discovering what makes you unique.",
            "It's a pleasure to meet you. Let's begin exploring your personality together.",
            "I'm glad you're here! Let's uncover your strengths and unique qualities."
        ]
        arabic_templates = [
            "يسعدني جدًا لقاؤك. أنا متحمس لمساعدتك في اكتشاف سماتك الشخصية الفريدة وما يجعلك مميزًا.",
            "مرحبًا! أنا هنا لمساعدتك في استكشاف سماتك الشخصية وقدراتك الفريدة.",
            "أهلًا وسهلًا! يسعدني أن أساعدك في اكتشاف ما يميزك.",
            "يشرفني لقاؤك. دعنا نبدأ معًا في استكشاف شخصيتك.",
            "سعيد بوجودك هنا! لنكتشف معًا نقاط قوتك وسماتك الفريدة."
        ]

        # Simple greeting/off-topic detection (pattern-based)
        text_lower = user_input.lower().strip()
        if language.lower() in ["ar", "arabic"]:
            greetings = ["مرحبا", "اهلا", "السلام عليكم", "صباح الخير", "مساء الخير", "هلا", "اهلين", "حياك", "كيف حالك", "كيفك", "شلونك", "يسعد صباحك", "يسعد مساءك", "تسلم", "نهارك سعيد"]
            for greeting in greetings:
                if greeting in text_lower:
                    response = arabic_templates[0]
                    return PersonalityAnalyzer.remove_questions(response)
            # If not a greeting, return empty string
            return ""
        else:
            greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you", "how do you do", "nice to meet you", "pleased to meet you", "greetings", "howdy", "what's up", "how's it going"]
            for greeting in greetings:
                if greeting in text_lower:
                    response = english_templates[0]
                    return PersonalityAnalyzer.remove_questions(response)
            # If not a greeting, return empty string
            return ""
    
    @staticmethod
    def get_varied_offtopic_response(user_input: str, language: str = "en", openai_client=None) -> str:
        """
        Generate varied, casual responses for off-topic content without emojis.
        Returns a short, conversational response that redirects to the main task.
        """
        if not user_input.strip():
            return ""
            
        if not openai_client:
            from openai import OpenAI
            openai_client = OpenAI()

        # Short, formal responses WITHOUT questions - suitable for concatenation
        english_templates = [
            "I understand your question, but that's outside my area of expertise. Let me help you with personality analysis instead.",
            "That's an interesting topic, however I specialize in personality analysis.",
            "I appreciate your curiosity, but I'm designed to focus on personality assessment.",
            "That's beyond my current scope, but I'd be happy to continue analyzing your personality traits.",
            "I recognize your interest in that topic, though my expertise is in personality analysis.",
            "While that's a fascinating subject, my role is to help with personality evaluation.",
            "I understand your question, but I'm specifically designed for personality analysis.",
            "That topic is outside my specialization, but I'm here to assist with your personality assessment.",
            "I see what you're asking about, however my focus is on personality analysis.",
            "That's not within my area of expertise, but I can certainly help you understand your personality better."
        ]
        
        arabic_templates = [
            "أفهم سؤالك، لكن هذا خارج مجال خبرتي. دعني أساعدك في تحليل الشخصية بدلاً من ذلك.",
            "هذا موضوع مثير للاهتمام، لكنني متخصص في تحليل الشخصية.",
            "أقدر فضولك، لكنني مصمم للتركيز على تقييم الشخصية.",
            "هذا خارج نطاقي الحالي، لكنني سأكون سعيداً لمتابعة تحليل سمات شخصيتك.",
            "أدرك اهتمامك بذلك الموضوع، لكن خبرتي في تحليل الشخصية.",
            "رغم أن هذا موضوع رائع، دوري هو المساعدة في تقييم الشخصية.",
            "أفهم سؤالك، لكنني مصمم خصيصاً لتحليل الشخصية.",
            "هذا الموضوع خارج تخصصي، لكنني هنا لمساعدتك في تقييم شخصيتك.",
            "أرى ما تسأل عنه، لكن تركيزي على تحليل الشخصية.",
            "هذا ليس في مجال خبرتي، لكن يمكنني بالتأكيد مساعدتك على فهم شخصيتك بشكل أفضل."
        ]

        system_prompt = f"""You are a professional personality analyst that provides formal, complete responses for off-topic queries.

MISSION: If the input is off-topic or not related to personality analysis, respond with a polite, professional redirect that flows well when concatenated with clarification questions.

RESPONSE STYLE:
- Use formal, complete sentences
- Be polite and professional
- NO emojis or casual expressions  
- NO QUESTIONS in your response
- Redirect professionally to personality analysis
- Make responses that flow well with follow-up questions
- Use the language: {"Arabic" if language == "ar" else "English"}
- End statements (not questions) that connect well to clarification questions

RESPONSE EXAMPLES for {"Arabic" if language == "ar" else "English"}:
{chr(10).join(f"- {template}" for template in (arabic_templates if language == "ar" else english_templates)[:5])}

RULES:
1. If off-topic: Generate ONE professional redirect response similar to examples
2. If personality-related: Return "none"  
3. Make it formal and complete for better concatenation
4. NO QUESTIONS - only statements that work well before clarification questions
5. Keep responses short and professional"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Respond to: '{user_input}'"}
                ],
                max_tokens=100,
                temperature=0.6
            )
            
            result = response.choices[0].message.content.strip()
            
            if result.lower() == "none":
                return ""
            else:
                return result
                
        except Exception as e:
            print(f"[LOG] Error in varied off-topic detection: {e}")
            # Fallback to random template
            import random
            templates = arabic_templates if language == "ar" else english_templates
            return random.choice(templates)
    
    @staticmethod
    def _create_combined_identity_response(categories: list, language: str) -> str:
        """
        Create a single, flowing, friendly response block that combines multiple identity categories
        into one readable, conversational sentence.
        """
        if not categories:
            return ""
        
        # Remove duplicates and maintain order
        unique_categories = []
        for cat in categories:
            if cat not in unique_categories:
                unique_categories.append(cat)
        
        if language == "ar":
            return PersonalityAnalyzer._create_arabic_flowing_response(unique_categories)
        else:
            return PersonalityAnalyzer._create_english_flowing_response(unique_categories)
    
    @staticmethod
    def _create_english_flowing_response(categories: list) -> str:
        """
        Create a single flowing English response that naturally combines all requested information.
        """
        parts = []
        
        # Start with friendly introduction
        if 'who_are_you' in categories:
            parts.append("I'm Minus Zero, your friendly AI assistant and part of the BEGINING project")
        
        # Add project description if asked
        if 'what_is_begining' in categories:
            if parts:
                parts.append("BEGINING is a comprehensive personality trait measurement system that explores the foundations of intellectual, behavioral, and societal excellence, classifying individuals into 120 unique personality types")
            else:
                parts.append("BEGINING is a comprehensive personality trait measurement system that explores intellectual, behavioral, and societal excellence")
        
        # Add purpose naturally
        if 'purpose' in categories:
            if parts:
                parts.append("My purpose is to guide you in discovering your unique strengths, patterns, and inclinations so you can better understand yourself and how you interact with the world around you")
            else:
                parts.append("My purpose is to guide you in discovering your strengths, patterns, and inclinations for better self-understanding")
        
        # Add role/methodology
        if 'role' in categories or 'how_analyze' in categories:
            if parts:
                parts.append("I analyze your personality using a structured approach that examines your emotional, social, cognitive, and behavioral traits to create your personalized profile")
            else:
                parts.append("I analyze personality through a structured approach examining emotional, social, cognitive, and behavioral traits")
        
        # Add team/developer info
        if 'developer' in categories or 'team' in categories:
            if parts:
                parts.append("I was developed by a talented team of Saudi researchers and engineers specializing in psychology, sociology, and artificial intelligence")
            else:
                parts.append("Developed by Saudi experts in psychology and AI technology")
        
        # Add objectives if specifically asked
        if 'objectives' in categories:
            if parts:
                parts.append("Our objectives include educational guidance, human resource development, academic research, and future integration with advanced AI systems")
            else:
                parts.append("We focus on educational guidance, HR development, and research applications")
        
        # Create one flowing sentence
        if len(parts) == 1:
            response = parts[0] + ". Let's begin uncovering what makes you unique!"
        elif len(parts) == 2:
            response = f"{parts[0]}, and {parts[1].lower()}. Let's begin this exciting journey of self-discovery together!"
        else:
            # Multiple parts - create flowing narrative
            response = f"{parts[0]}. {parts[1]}, and {' '.join(parts[2:]).lower()}. I'm here to help you explore your traits, tendencies, and inner potential. Let's begin uncovering what makes you truly unique!"
        
        return response
    
    @staticmethod
    def _create_arabic_flowing_response(categories: list) -> str:
        """
        Create a single flowing Arabic response that naturally combines all requested information.
        """
        parts = []
        
        # Start with friendly introduction
        if 'who_are_you' in categories:
            parts.append("أنا ماينس زيرو، مساعدك الذكي الودود وجزء من مشروع BEGINING")
        
        # Add project description
        if 'what_is_begining' in categories:
            if parts:
                parts.append("BEGINING هو نظام شامل لقياس سمات الشخصية يستكشف أسس التميز الفكري والسلوكي والاجتماعي، ويصنف الأفراد إلى 120 نوعًا فريدًا من الشخصيات")
            else:
                parts.append("BEGINING هو نظام شامل لقياس سمات الشخصية يستكشف التميز الفكري والسلوكي")
        
        # Add purpose
        if 'purpose' in categories:
            if parts:
                parts.append("هدفي هو إرشادك لاكتشاف نقاط قوتك وأنماطك وميولك الفريدة، لتتمكن من فهم نفسك بشكل أفضل وطريقة تفاعلك مع العالم من حولك")
            else:
                parts.append("هدفي إرشادك لاكتشاف نقاط قوتك وأنماطك لفهم أفضل لذاتك")
        
        # Add role/methodology
        if 'role' in categories or 'how_analyze' in categories:
            if parts:
                parts.append("أحلل شخصيتك باستخدام منهج منظم يدرس السمات العاطفية والاجتماعية والمعرفية والسلوكية لإنشاء ملفك الشخصي المخصص")
            else:
                parts.append("أحلل الشخصية عبر منهج منظم يدرس السمات العاطفية والاجتماعية والمعرفية")
        
        # Add team/developer info
        if 'developer' in categories or 'team' in categories:
            if parts:
                parts.append("تم تطويري من قبل فريق موهوب من الباحثين والمهندسين السعوديين المتخصصين في علم النفس وعلم الاجتماع والذكاء الاصطناعي")
            else:
                parts.append("طُورت من قبل خبراء سعوديين في علم النفس وتقنية الذكاء الاصطناعي")
        
        # Add objectives if asked
        if 'objectives' in categories:
            if parts:
                parts.append("أهدافنا تشمل الإرشاد التعليمي وتطوير الموارد البشرية والبحث الأكاديمي والتكامل المستقبلي مع أنظمة الذكاء الاصطناعي المتقدمة")
            else:
                parts.append("نركز على الإرشاد التعليمي وتطوير الموارد البشرية والتطبيقات البحثية")
        
        # Create one flowing response
        if len(parts) == 1:
            response = parts[0] + ". لنبدأ باكتشاف ما يميزك!"
        elif len(parts) == 2:
            response = f"{parts[0]}، و{parts[1]}. لنبدأ هذه الرحلة المثيرة لاكتشاف الذات معًا!"
        else:
            # Multiple parts - create flowing narrative  
            response = f"{parts[0]}. {parts[1]}، و{' '.join(parts[2:])}. أنا هنا لمساعدتك على استكشاف سماتك وميولك وإمكاناتك الداخلية. لنبدأ باكتشاف ما يجعلك مميزًا حقًا!"
        
        return response
        
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
IMPORTANT: The personal_greeting may now contain both greeting and clarification question combined in one natural response.
- When personal_greeting contains a combined greeting+question, use it as-is and set clarification_questions to empty array
- When personal_greeting is just a greeting, you may generate separate clarification_questions
- This creates more natural conversation flow by connecting greetings with relevant questions

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
    "description_identity": null,
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

    @staticmethod
    def _extract_traits_by_pattern(text: str) -> set:
        """
        Extract personality traits using regex patterns as a fallback.
        Returns a set of trait categories found in the text.
        Supports both English and Arabic text.
        """
        import re
        found_traits = set()
        text_lower = text.lower()
        
        # English patterns
        english_patterns = {
            "emotional": r"enthusiastic|happy|sad|calm|feel(s)?|emotion|stress|excited|passion|motivat(ed|ion)|anxiety|angry|nervous|worried|content|optimistic|pessimistic|joyful|frustrated|relaxed|overwhelmed|mood|temper|patient|sensitive|expressive|reserved|emotional intelligence|cope|satisfaction|proud|embarrassed|guilty|inspired|enjoy|love|like|hate|dislike",
            "social": r"collaborative|team|help|assist|shy|introvert|extrovert|interact|polite|friendly|people|others|social|network|connection|relationship|communicate|listen|leadership|followership|assertive|passive|aggressive|empathy|sympathy|understand|socialize|negotiate|persuade|influence|charm|crowd|isolation|community|group|belong|inclusion|exclusion|trust|distrust|approachable|distant|boundary|conflict|mentor|colleagues|discussions",
            "cognitive": r"think|critical|logical|analytical|understand|reason|solve|strateg(y|ic)|intuitive|creative|innovative|practical|abstract|concrete|detail-oriented|big picture|conceptual|perspective|mental|intellectual|curious|learning|knowledge|information|decision|judgment|bias|objective|subjective|rational|irrational|memory|attention|focus|concentrate|distracted|multi-task|prioritize|plan|reflect|comprehend|insight|wisdom|intelligence|data|analysis|patterns|insights|problems",
            "behavioral": r"organized|spontaneous|routine|habit|act|impulsive|disciplined|methodical|child(ish)?|consistent|reliable|flexible|rigid|adaptable|predictable|unpredictable|responsible|irresponsible|cautious|risk-taking|procrastinate|proactive|reactive|efficient|systematic|messy|neat|punctual|late|deadline|priority|goal|achievement|motivation|ambition|lazy|industrious|perseverance|persistence|give up|determined|stubborn|exercise|diet|sleep|activity|energetic|sedentary|roles|working"
        }
        
        # Arabic patterns
        arabic_patterns = {
            "emotional": r"يستمتع|أستمتع|أحب|يحب|أشعر|يشعر|رضا|سعيد|حزين|هادئ|متحمس|شغوف|قلق|غاضب|متوتر|قلق|راض|متفائل|متشائم|فرح|محبط|مسترخي|مرهق|مزاج|صبور|حساس|معبر|محفوظ|ذكي عاطفي|تأقلم|رضا|فخور|محرج|مذنب|ملهم|استمتاع|حب|إعجاب|كراهية|عدم إعجاب",
            "social": r"تعاوني|فريق|فرق|مساعدة|يساعد|خجول|منطوي|منفتح|تفاعل|مهذب|ودود|الناس|الآخرين|اجتماعي|شبكة|اتصال|علاقة|تواصل|استماع|قيادة|قائد|أدوار القيادة|حازم|سلبي|عدواني|تعاطف|تفهم|اجتماع|تفاوض|إقناع|تأثير|سحر|حشد|عزلة|مجتمع|مجموعة|انتماء|شمول|استبعاد|ثقة|عدم ثقة|ودود|بعيد|حدود|صراع|إرشاد|زملاء|مناقشات",
            "cognitive": r"تفكير|نقدي|منطقي|تحليلي|فهم|سبب|حل|استراتيجي|بديهي|إبداعي|مبتكر|عملي|مجرد|ملموس|موجه للتفاصيل|الصورة الكبيرة|مفاهيمي|منظور|عقلي|فكري|فضولي|تعلم|معرفة|معلومات|قرار|حكم|تحيز|موضوعي|ذاتي|عقلاني|غير عقلاني|ذاكرة|انتباه|تركيز|تركيز|مشتت|متعدد المهام|أولوية|خطة|تأمل|فهم|بصيرة|حكمة|ذكاء|بيانات|تحليل|أنماط|رؤى|مشاكل|مشكلات",
            "behavioral": r"منظم|عفوي|روتين|عادة|فعل|متهور|منضبط|منهجي|طفولي|متسق|موثوق|مرن|جامد|قابل للتكيف|متوقع|غير متوقع|مسؤول|غير مسؤول|حذر|مخاطر|تأجيل|استباقي|رد فعل|فعال|منهجي|فوضوي|أنيق|دقيق|متأخر|موعد نهائي|أولوية|هدف|إنجاز|دافع|طموح|كسلان|مجتهد|مثابرة|إصرار|استسلام|مصمم|عنيد|تمرين|نظام غذائي|نوم|نشاط|نشيط|خامل|أدوار|عمل|عامل"
        }
        
        # Check English patterns
        for trait_category, pattern in english_patterns.items():
            if re.search(pattern, text_lower):
                found_traits.add(trait_category)
        
        # Check Arabic patterns (no need to lowercase for Arabic)
        for trait_category, pattern in arabic_patterns.items():
            if re.search(pattern, text):
                found_traits.add(trait_category)
        
        return found_traits

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

        # 1. Check for conversation context first
        has_conversation_history = bool(new_input and isinstance(new_input, list) and len(new_input) > 0)
        
        # 2. For conversation with history: check ONLY the LATEST answer
        if has_conversation_history:
            latest_qa = new_input[-1]  # Get the last (most recent) Q&A pair
            latest_answer = latest_qa.get("answer", "").strip()
            self.logger.info(f"Step: Conversation detected, analyzing LATEST answer: {latest_answer}")
            
            if latest_answer:
                # Check latest answer in order: greeting → identity → off-topic → personality
                
                # 1. Check if latest answer is greeting (highest priority)
                try:
                    greeting_response = self.get_greeting_or_offtopic_response(latest_answer, detected_languages, openai_client=self.client)
                    if greeting_response and self._is_greeting(latest_answer, detected_languages):
                        self.logger.info(f"Step: Latest answer is greeting")
                        contextual_question = "هل يمكنك أن تخبرني كيف تتفاعل مع الآخرين في المواقف الاجتماعية؟" if detected_languages == "ar" else "Could you tell me how you typically interact with others in social situations?"
                        
                        result = {
                            "id": id,
                            "status": "incomplete",
                            "personal_greeting_and_off_topic": greeting_response,
                            "description_english": "",
                            "description_arabic": "",
                            "description_identity": None,
                            "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                            "clarification_questions": [contextual_question],
                            "input_tokens": len(user_input.split()),
                            "output_tokens": len(greeting_response.split()),
                            "total_tokens": len(user_input.split()) + len(greeting_response.split())
                        }
                        self.logger.info(f"[EXIT] analyze (latest answer greeting): {result}")
                        self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
                        return result
                except Exception as e:
                    self.logger.error(f"Error checking latest answer for greeting: {e}")
                
                # 2. Check if latest answer is identity question
                try:
                    identity_response = self.get_identity_response(latest_answer, detected_languages, openai_client=self.client)
                    if identity_response:
                        self.logger.info(f"Step: Latest answer is identity question")
                        # Generate clarification question for personality traits
                        try:
                            clarification_questions = self.generate_clarification_questions_gpt(["emotional", "social", "cognitive", "behavioral"], detected_languages, openai_client=self.client, logger=self.logger)
                        except Exception as e:
                            self.logger.error(f"Error generating clarification questions: {e}")
                            clarification_questions = []
                        if not clarification_questions:
                            contextual_question = "هل يمكنك أن تخبرني كيف تتفاعل مع الآخرين في المواقف الاجتماعية؟" if detected_languages == "ar" else "Could you tell me how you typically interact with others in social situations?"
                            clarification_questions = [contextual_question]
                        
                        result = {
                            "id": id,
                            "status": "incomplete",
                            "personal_greeting_and_off_topic": "",
                            "description_english": "",
                            "description_arabic": "",
                            "description_identity": identity_response,
                            "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                            "clarification_questions": clarification_questions,
                            "input_tokens": len(user_input.split()),
                            "output_tokens": len(identity_response.split()),
                            "total_tokens": len(user_input.split()) + len(identity_response.split())
                        }
                        self.logger.info(f"[EXIT] analyze (latest answer identity): {result}")
                        self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
                        return result
                except Exception as e:
                    self.logger.error(f"Error checking latest answer for identity: {e}")
                
                # 3. Check if latest answer is off-topic
                try:
                    off_topic_response = self.get_greeting_or_offtopic_response(latest_answer, detected_languages, openai_client=self.client)
                    if off_topic_response and not self._is_greeting(latest_answer, detected_languages):
                        self.logger.info(f"Step: Latest answer is off-topic (direct detection), generating varied response")
                        # Generate varied casual response
                        try:
                            varied_response = self.get_varied_offtopic_response(latest_answer, detected_languages, openai_client=self.client)
                        except Exception as e:
                            self.logger.error(f"Error generating varied off-topic response: {e}")
                            if detected_languages == "ar":
                                varied_response = "تمام لكن هذا خارج نطاقي. هل نكمل طلبك؟"
                            else:
                                varied_response = "Got it but that's outside my scope. Want me to continue your request?"
                        
                        contextual_question = "هل يمكنك أن تخبرني كيف تتفاعل مع الآخرين في المواقف الاجتماعية؟" if detected_languages == "ar" else "Could you tell me how you typically interact with others in social situations?"
                        
                        result = {
                            "id": id,
                            "status": "incomplete",
                            "personal_greeting_and_off_topic": varied_response,
                            "description_english": "",
                            "description_arabic": "",
                            "description_identity": None,
                            "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                            "clarification_questions": [contextual_question],
                            "input_tokens": len(user_input.split()),
                            "output_tokens": len(varied_response.split()),
                            "total_tokens": len(user_input.split()) + len(varied_response.split())
                        }
                        self.logger.info(f"[EXIT] analyze (latest answer off-topic): {result}")
                        self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
                        return result
                    else:
                        # Check if answer has no personality content (alternative off-topic detection)
                        personality_content = self._extract_personality_content_from_mixed_input(latest_answer, detected_languages)
                        if not personality_content.strip():
                            self.logger.info(f"Step: Latest answer is off-topic (no personality content), generating varied response")
                            # Generate varied casual response
                            try:
                                varied_response = self.get_varied_offtopic_response(latest_answer, detected_languages, openai_client=self.client)
                            except Exception as e:
                                self.logger.error(f"Error generating varied off-topic response: {e}")
                                if detected_languages == "ar":
                                    varied_response = "تمام لكن هذا خارج نطاقي. هل نكمل طلبك؟"
                                else:
                                    varied_response = "Got it but that's outside my scope. Want me to continue your request?"
                            
                            contextual_question = "هل يمكنك أن تخبرني كيف تتفاعل مع الآخرين في المواقف الاجتماعية؟" if detected_languages == "ar" else "Could you tell me how you typically interact with others in social situations?"
                            
                            result = {
                                "id": id,
                                "status": "incomplete",
                                "personal_greeting_and_off_topic": varied_response,
                                "description_english": "",
                                "description_arabic": "",
                                "description_identity": None,
                                "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                                "clarification_questions": [contextual_question],
                                "input_tokens": len(user_input.split()),
                                "output_tokens": len(varied_response.split()),
                                "total_tokens": len(user_input.split()) + len(varied_response.split())
                            }
                            self.logger.info(f"[EXIT] analyze (latest answer off-topic via personality check): {result}")
                            self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
                            return result
                except Exception as e:
                    self.logger.error(f"Error checking latest answer for off-topic: {e}")
                
                # 4. If latest answer is none of the above, continue with personality analysis
                self.logger.info(f"Step: Latest answer is personality-related, continuing with trait analysis")
                
        # Clean user_input to contain only self-description content for personality analysis
        self.logger.info(f"Step: Cleaning user_input to extract only personality content")
        cleaned_user_input = self._extract_personality_content_from_mixed_input(user_input, detected_languages)
        if cleaned_user_input.strip():
            self.logger.info(f"Step: Cleaned user_input for personality analysis: {cleaned_user_input}")
            user_input = cleaned_user_input.strip()
        else:
            self.logger.info(f"Step: No personality content found in user_input after cleaning - treating as off-topic")
            # When no personality content is found, treat as off-topic
            try:
                varied_response = self.get_varied_offtopic_response(user_input, detected_languages, openai_client=self.client)
            except Exception as e:
                self.logger.error(f"Error generating varied off-topic response: {e}")
                if detected_languages == "ar":
                    varied_response = "أفهم سؤالك، لكن هذا خارج مجال خبرتي. دعني أساعدك في تحليل الشخصية بدلاً من ذلك."
                else:
                    varied_response = "I understand your question, but that's outside my area of expertise. Let me help you with personality analysis instead."
            
            # Generate clarification question for personality traits
            try:
                clarification_questions = self.generate_clarification_questions_gpt(["emotional", "social", "cognitive", "behavioral"], detected_languages, openai_client=self.client, logger=self.logger)
            except Exception as e:
                self.logger.error(f"Error generating clarification questions: {e}")
                clarification_questions = []
            if not clarification_questions:
                contextual_question = "هل يمكنك أن تخبرني كيف تتفاعل مع الآخرين في المواقف الاجتماعية؟" if detected_languages == "ar" else "Could you tell me how you typically interact with others in social situations?"
                clarification_questions = [contextual_question]
            
            result = {
                "id": id,
                "status": "incomplete",
                "personal_greeting_and_off_topic": varied_response,
                "description_english": "",
                "description_arabic": "",
                "description_identity": None,
                "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                "clarification_questions": clarification_questions,
                "input_tokens": len(user_input.split()),
                "output_tokens": len(varied_response.split()),
                "total_tokens": len(user_input.split()) + len(varied_response.split())
            }
            self.logger.info(f"[EXIT] analyze (user_input off-topic - no personality content): {result}")
            self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
            return result
        
        # Continue with the rest of the analysis...
            
            # Check user_input for off-topic/greeting
            try:
                greeting_response = self.get_greeting_or_offtopic_response(user_input, detected_languages, openai_client=self.client)
                if greeting_response:
                    self.logger.info(f"Step: user_input is greeting/off-topic")
                    contextual_question = "هل يمكنك أن تخبرني المزيد عن نفسك؟" if detected_languages == "ar" else "Could you tell me more about yourself?"
                    
                    result = {
                        "id": id,
                        "status": "incomplete",
                        "personal_greeting_and_off_topic": greeting_response,
                        "description_english": "",
                        "description_arabic": "",
                        "description_identity": None,
                        "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                        "clarification_questions": [contextual_question],
                        "input_tokens": len(user_input.split()),
                        "output_tokens": len(greeting_response.split()),
                        "total_tokens": len(user_input.split()) + len(greeting_response.split())
                    }
                    self.logger.info(f"[EXIT] analyze (user_input greeting/off-topic): {result}")
                    self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
                    return result
            except Exception as e:
                self.logger.error(f"Error checking user_input for greeting/off-topic: {e}")
            
            # Check user_input for identity
            try:
                identity_response = self.get_identity_response(user_input, detected_languages, openai_client=self.client)
                if identity_response:
                    self.logger.info(f"Step: user_input is identity question")
                    try:
                        clarification_questions = self.generate_clarification_questions_gpt(["emotional", "social", "cognitive", "behavioral"], detected_languages, openai_client=self.client, logger=self.logger)
                    except Exception as e:
                        self.logger.error(f"Error generating clarification questions: {e}")
                        clarification_questions = []
                    if not clarification_questions:
                        contextual_question = "هل يمكنك أن تخبرني كيف تتفاعل مع الآخرين في المواقف الاجتماعية؟" if detected_languages == "ar" else "Could you tell me how you typically interact with others in social situations?"
                        clarification_questions = [contextual_question]
                    
                    result = {
                        "id": id,
                        "status": "incomplete",
                        "personal_greeting_and_off_topic": "",
                        "description_english": "",
                        "description_arabic": "",
                        "description_identity": identity_response,
                        "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                        "clarification_questions": clarification_questions,
                        "input_tokens": len(user_input.split()),
                        "output_tokens": len(identity_response.split()),
                        "total_tokens": len(user_input.split()) + len(identity_response.split())
                    }
                    self.logger.info(f"[EXIT] analyze (user_input identity): {result}")
                    self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
                    return result
            except Exception as e:
                self.logger.error(f"Error checking user_input for identity: {e}")

        # 3. Continue with personality trait extraction
        trait_patterns = self.TRAIT_PATTERNS
        present_traits = set()
        answers_for_traits = []
        identity_response = ""

        # Check if user_input contains identity questions first
        is_user_input_identity = False
        try:
            user_identity_response = self.get_identity_response(user_input, detected_languages, openai_client=self.client)
            if user_identity_response:
                identity_response = user_identity_response
                is_user_input_identity = True
                self.logger.info(f"Step: user_input detected as identity question")
        except Exception as e:
            self.logger.error(f"Error in identity detection for user_input: {e}")
            
        # Check if the LATEST answer in new_input is an identity question (for identity responses)
        latest_answer = None
        if not is_user_input_identity and new_input and isinstance(new_input, list):
            latest_answer = new_input[-1].get("answer", "").strip()
        self.logger.info(f"Step: Latest answer in new_input (for identity response): {latest_answer}")
        is_latest_identity = False
        if not is_user_input_identity:
            try:
                if latest_answer and self.get_identity_response(latest_answer, detected_languages, openai_client=self.client):
                    identity_response = self.get_identity_response(latest_answer, detected_languages, openai_client=self.client)
                    is_latest_identity = True
                elif not identity_response:
                    identity_response = ""
            except Exception as e:
                self.logger.error(f"Error in identity detection for latest_answer: {e}")
                if not identity_response:
                    identity_response = ""
        self.logger.info(f"Step: Identity response: {identity_response}")
        self.logger.info(f"Step: is_latest_identity={is_latest_identity}, is_user_input_identity={is_user_input_identity}")

        # ALWAYS include user_input for trait extraction, but clean it of identity questions
        # Handle mixed content by separating personality descriptions from identity questions
        cleaned_user_input = self._extract_personality_content_from_mixed_input(user_input, detected_languages)
        if cleaned_user_input.strip():
            answers_for_traits.append(cleaned_user_input)
            self.logger.info(f"Step: Added user_input to trait extraction (cleaned)")
        else:
            self.logger.info(f"Step: user_input contained only identity questions, skipped")

        # Accumulate ONLY personality descriptions from conversation history
        # Filter out identity answers and only use self-descriptions for personality analysis
        # Process in chronological order but prioritize LATEST answers for trait extraction
        identity_answers_found = []
        off_topic_answers_found = []
        
        for i, qa in enumerate(new_input or []):
            answer = qa.get("answer", "").strip()
            if not answer:
                continue
                
            self.logger.info(f"Step: Checking answer {i+1} for personality content: {answer}")
            try:
                # Check if this answer is an identity question
                is_identity = bool(self.get_identity_response(answer, detected_languages, openai_client=self.client))
                
                if is_identity:
                    identity_answers_found.append((i, answer))
                    self.logger.info(f"Step: Skipped answer {i+1} (detected as identity question): {answer}")
                else:
                    # Check if this answer is off-topic/greeting
                    off_topic_response = self.get_greeting_or_offtopic_response(answer, detected_languages, openai_client=self.client)
                    if off_topic_response:
                        off_topic_answers_found.append((i, answer))
                        self.logger.info(f"Step: Answer {i+1} detected as off-topic: {answer}")
                    else:
                        # Extract personality content from non-identity, non-off-topic answers
                        personality_content = self._extract_personality_content_from_mixed_input(answer, detected_languages)
                        if personality_content.strip():
                            answers_for_traits.append(personality_content)
                            self.logger.info(f"Step: Added answer {i+1} to trait extraction (personality content): {personality_content}")
                        else:
                            # If no personality content and not explicitly off-topic, treat as off-topic
                            off_topic_answers_found.append((i, answer))
                            self.logger.info(f"Step: Answer {i+1} contained no personality content, treating as off-topic")
                    
            except Exception as e:
                self.logger.error(f"Error processing answer {i+1} for personality content: {e}")
                # If error, treat as potential personality content
                answers_for_traits.append(answer)
        
        # Log identity and off-topic answers found for debugging
        if identity_answers_found:
            self.logger.info(f"Step: Identity answers found at positions: {[pos for pos, _ in identity_answers_found]}")
        if off_topic_answers_found:
            self.logger.info(f"Step: Off-topic answers found at positions: {[pos for pos, _ in off_topic_answers_found]}")
        
        # Check if all new_input answers are off-topic (and no personality content from user_input)
        if new_input and off_topic_answers_found and len(off_topic_answers_found) == len(new_input) and not answers_for_traits:
            self.logger.info(f"Step: All conversation answers are off-topic, generating off-topic response")
            
            # Generate a varied, casual off-topic response
            try:
                # Combine all off-topic answers to generate a contextual response
                all_off_topic_content = user_input + " " + " ".join([answer for _, answer in off_topic_answers_found])
                off_topic_combined_response = self.get_varied_offtopic_response(all_off_topic_content, detected_languages, openai_client=self.client)
            except Exception as e:
                self.logger.error(f"Error generating varied off-topic response: {e}")
                # Fallback to simple template
                if detected_languages == "ar":
                    off_topic_combined_response = "تمام لكن هذا خارج نطاقي. هل نكمل طلبك؟"
                else:
                    off_topic_combined_response = "Got it but that's outside my scope. Want me to continue your request?"
            
            # Generate appropriate follow-up question
            if detected_languages == "ar":
                contextual_question = "هل يمكنك أن تخبرني كيف تتفاعل مع الآخرين في المواقف الاجتماعية؟"
            else:
                contextual_question = "Could you tell me how you typically interact with others in social situations?"
            
            result = {
                "id": id,
                "status": "incomplete",
                "personal_greeting_and_off_topic": off_topic_combined_response,
                "description_english": "",
                "description_arabic": "",
                "description_identity": None,
                "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
                "clarification_questions": [contextual_question],
                "input_tokens": len(user_input.split()),
                "output_tokens": len(off_topic_combined_response.split()) if off_topic_combined_response else 0,
                "total_tokens": len(user_input.split()) + (len(off_topic_combined_response.split()) if off_topic_combined_response else 0)
            }
            self.logger.info(f"[EXIT] analyze (all off-topic): {result}")
            self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
            return result
        
        self.logger.info(f"Step: Final answers for trait extraction: {answers_for_traits}")

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
                gpt_result = {"detected_traits": []}
            
            # Always use pattern matching as supplementary detection
            self.logger.info(f"Step: Using supplementary pattern matching for answer: {answer}")
            pattern_traits = self._extract_traits_by_pattern(answer)
            self.logger.info(f"Step: Pattern-based traits for '{answer}': {pattern_traits}")
            for trait in pattern_traits:
                present_traits.add(trait)
                    
        # Always check for all four traits
        all_traits = set(["emotional", "social", "cognitive", "behavioral"])
        missing_traits = [trait for trait in all_traits if trait not in present_traits]
        self.logger.info(f"Step: Present traits: {present_traits}, Missing traits: {missing_traits}")

        # 3. Output logic - Priority order: Complete personality > Identity > Incomplete

        # FIRST: Check if personality analysis is complete (all traits found)
        if not missing_traits:
            # All traits present, generate complete personality description
            # Use GPT to create a flowing description from all personality content
            personality_text = " ".join(answers_for_traits)
            
            try:
                # Generate professional personality description
                description_prompt = f"""Based on the following personality information, create a professional, flowing personality description that captures the person's key traits:

{personality_text}

Create a 2-3 sentence description that highlights their emotional, social, cognitive, and behavioral characteristics in a natural, professional manner."""

                description_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional personality analyst. Create flowing, professional personality descriptions."},
                        {"role": "user", "content": description_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                
                generated_description = description_response.choices[0].message.content.strip()
                
            except Exception as e:
                self.logger.error(f"Error generating personality description: {e}")
                # Fallback to simple concatenation
                generated_description = personality_text
            
            result = {
                "id": id,
                "status": "complete",
                "personal_greeting_and_off_topic": "",
                "description_english": generated_description if detected_languages == "en" else "",
                "description_arabic": generated_description if detected_languages == "ar" else "",
                "description_identity": None,
                "missing_traits": [],
                "clarification_questions": [],
                "input_tokens": len(user_input.split()),
                "output_tokens": len(generated_description.split()) if generated_description else 0,
                "total_tokens": len(user_input.split()) + (len(generated_description.split()) if generated_description else 0)
            }
            self.logger.info(f"[EXIT] analyze (complete personality): {result}")
            self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
            return result

        # SECOND: Check if identity response should be returned (when traits are incomplete)
        if is_latest_identity or is_user_input_identity:
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
                "description_identity": identity_response if identity_response else None,  # Only set if identity detected
                "missing_traits": missing_traits,
                "clarification_questions": clarification_questions,
                "input_tokens": len(user_input.split()),
                "output_tokens": len(identity_response.split()),
                "total_tokens": len(user_input.split()) + len(identity_response.split())
            }
            self.logger.info(f"[EXIT] analyze (identity): {result}")
            self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
            return result

        # THIRD: Handle incomplete personality analysis (no identity questions but missing traits)
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
            "description_identity": None,  # Only set if identity detected
            "missing_traits": missing_traits,
            "clarification_questions": clarification_questions,
            "input_tokens": len(user_input.split()),
            "output_tokens": 0,
            "total_tokens": len(user_input.split())
        }
        self.logger.info(f"[EXIT] analyze (incomplete): {result}")
        self.logger.info(f"Step: Duration: {time.time() - start_time:.3f}s")
        return result
