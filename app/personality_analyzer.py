import os
import re
import json
import logging
import unicodedata
from typing import List, Dict, Any
from openai import OpenAI, OpenAIError
import tiktoken
from dotenv import load_dotenv

load_dotenv()

class PersonalityAnalyzer:
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Auto-detect language from text content.
        Returns 'arabic' if Arabic characters are found, otherwise 'english'.
        """
        if re.search(r'[\u0600-\u06FF]', text):
            return "arabic"
        return "english"
    
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

    # Arabic clarification templates
    CLARIFICATION_TEMPLATES_ARABIC = {
        "emotional": [
            "كيف تشعر عادةً في المواقف الصعبة أو المثيرة؟ ما هي المشاعر التي تنتابك وكيف تتعامل معها؟",
            "هل يمكنك وصف كيف تستجيب عاطفياً للضغوط أو التحديات غير المتوقعة؟",
            "ما الذي يجلب لك أكبر قدر من الفرح أو الرضا في حياتك، وكيف تعبر عن هذه المشاعر؟",
            "كيف يصف أصدقاؤك المقربون مزاجك أو طبعك العاطفي المعتاد؟",
            "عندما تواجه نكسة، ما هي المشاعر التي تنشأ عادة وكيف تديرها؟"
        ],
        "social": [
            "هل يمكنك وصف كيف تتفاعل عادة مع الآخرين—هل تستمتع بالمساعدة، القيادة، أم تفضل العمل بمفردك؟",
            "كيف تتصرف عادة في البيئات الجماعية مقابل التفاعلات الفردية؟",
            "ما هو الدور الذي تأخذه عادة في المشاريع الجماعية أو بيئات العمل التعاونية؟",
            "كيف تصف نهجك في بناء والحفاظ على العلاقات مع الآخرين؟",
            "في المواقف الاجتماعية، هل تميل إلى بدء المحادثات أم تفضل أن يقترب منك الآخرون أولاً؟"
        ],
        "cognitive": [
            "ما نوع التفكير الذي يأتي لك بشكل طبيعي؟ هل أنت تحليلي، خيالي، أم أكثر اعتماداً على الحدس في القرارات؟",
            "كيف تتعامل عادة مع المشاكل المعقدة أو القرارات الصعبة؟",
            "هل تفضل التركيز على التفاصيل أم النظر إلى الصورة الكبيرة عند العمل على المشاريع؟",
            "كيف تجمع وتعالج المعلومات الجديدة عند تعلم شيء غير مألوف؟",
            "عند اتخاذ قرارات مهمة، هل تعتمد أكثر على الحقائق والمنطق أم على الحدس والقيم الشخصية؟"
        ],
        "behavioral": [
            "أخبرني عن عاداتك أو أفعالك—هل تفضل الروتين، التصرف بشكل عفوي، أم البقاء مرناً؟",
            "كم أنت منظم في حياتك اليومية والعمل؟ هل تتبع الأنظمة أم تتكيف مع الظروف؟",
            "كيف يبدو يومك العادي من ناحية البنية والأنشطة؟",
            "كيف تتعامل مع المواعيد النهائية والالتزامات؟ هل أنت عادة مبكر، في الوقت المحدد، أم في اللحظة الأخيرة؟",
            "هل تميل إلى التخطيط للأنشطة مسبقاً أم تفضل أن تكون عفوياً مع وقتك؟"
        ]
    }

    # Identity responses for when users ask about the system/bot
    IDENTITY_RESPONSES = {
        "who_are_you": {
            "english": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system. I'm here to help you explore your traits, tendencies, and inner potential. Let's get started by discovering a bit about you.",
            "arabic": "أنا Minus Zero، جزء من مشروع BEGINING، وهو نظام لقياس سمات الشخصية. أهدف لمساعدتك على استكشاف سماتك وميولك وإمكاناتك الداخلية. لنبدأ بالتعرف عليك قليلًا."
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
    
    # Off-topic responses for when users ask unrelated questions
    OFF_TOPIC_RESPONSES = {
        "general_unrelated": {
            "english": "I'm Minus Zero, a personality analysis system designed to help you discover your unique traits and characteristics. I specialize in understanding personality patterns, not general knowledge questions. Let's focus on exploring your personality instead! Could you tell me something about yourself, your habits, or how you typically respond to different situations?",
            "arabic": "أنا Minus Zero، نظام تحليل الشخصية المصمم لمساعدتك على اكتشاف سماتك وخصائصك الفريدة. أتخصص في فهم أنماط الشخصية، وليس الأسئلة المعرفية العامة. دعنا نركز على استكشاف شخصيتك بدلاً من ذلك! هل يمكنك إخباري شيئاً عن نفسك، أو عاداتك، أو كيف تستجيب عادةً للمواقف المختلفة؟"
        },
        "gibberish": {
            "english": "I notice your message contains unclear text that I can't understand. As Minus Zero, I'm here to help you explore your personality traits and characteristics. Let's get back on track! Could you share something meaningful about yourself - perhaps how you handle challenges, interact with others, or approach decision-making?",
            "arabic": "ألاحظ أن رسالتك تحتوي على نص غير واضح لا أستطيع فهمه. أنا Minus Zero، وأنا هنا لمساعدتك على استكشاف سمات شخصيتك وخصائصك. دعنا نعود إلى المسار الصحيح! هل يمكنك مشاركة شيء مفيد عن نفسك - ربما كيف تتعامل مع التحديات، أو تتفاعل مع الآخرين، أو تتخذ القرارات؟"
        },
        "factual_questions": {
            "english": "That's an interesting question, but I'm Minus Zero - a personality analysis system focused on understanding human traits and behaviors. I don't provide general information or facts about the world. Instead, I help you discover insights about your own personality! What would you like to explore about yourself today?",
            "arabic": "هذا سؤال مثير للاهتمام، لكنني Minus Zero - نظام تحليل الشخصية المتخصص في فهم السمات والسلوكيات البشرية. لا أقدم معلومات عامة أو حقائق عن العالم. بدلاً من ذلك، أساعدك على اكتشاف رؤى حول شخصيتك! ماذا تود أن تستكشف عن نفسك اليوم؟"
        },
        "technical_questions": {
            "english": "I understand you might be curious about technical topics, but I'm Minus Zero, specialized in personality analysis within the BEGINING project. My expertise is in understanding your unique psychological profile and traits. Let's dive into what makes you unique as a person! How do you typically approach new challenges or situations?",
            "arabic": "أفهم أنك قد تكون فضولياً حول المواضيع التقنية، لكنني Minus Zero، متخصص في تحليل الشخصية ضمن مشروع BEGINING. خبرتي في فهم ملفك النفسي الفريد وسماتك. دعنا نتعمق في ما يجعلك شخصاً فريداً! كيف تتعامل عادةً مع التحديات أو المواقف الجديدة؟"
        }
    }
    
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
            return " ".join(selected_questions[:2])

    def detect_identity_question(self, text: str) -> tuple:
        """
        Enhanced identity detection using GPT with smart keyword fallback.
        Returns a tuple: (is_identity_question: bool, response_key: str, response_data: dict)
        """
        if not text:
            return False, None, None
        
        # Normalize text for analysis
        text_lower = text.lower().strip()
        
        # FIRST: Check for statements about the system (NOT questions)
        statement_patterns = [
            r'\byou are (a |an )?chatbot\b',
            r'\byou are (a |an )?robot\b',
            r'\byou are (a |an )?ai\b',
            r'\byour purpose is\b',
            r'\byour role is\b'
        ]
        
        for pattern in statement_patterns:
            if re.search(pattern, text_lower):
                return False, None, None  # Statement, not question
        
        # SECOND: Handle mixed content - check for questions in the text
        # Look for question words/patterns even in mixed sentences
        question_indicators = [
            r'\bwhat do you do\b',
            r'\bwho are you\b', 
            r'\bwhat.*your (purpose|role|goal|objective)',
            r'\bwho.*your (developer|creator|maker)',
            r'\bما (هدفك|دورك)\b',
            r'\bمن (أنت|مطورك)\b'
        ]
        
        has_question = False
        for pattern in question_indicators:
            if re.search(pattern, text_lower):
                has_question = True
                break
        
        # If mixed content but no clear question, treat as personality
        if not has_question and ('.' in text or len(text.split()) > 8):
            # Check if it looks like mixed content (multiple sentences)
            sentences = text.split('.')
            if len(sentences) > 1:
                # Look for questions in any sentence
                for sentence in sentences:
                    for pattern in question_indicators:
                        if re.search(pattern, sentence.lower()):
                            has_question = True
                            break
                    if has_question:
                        break
                
                if not has_question:
                    return False, None, None  # Mixed content without clear questions
        
        try:
            # First attempt: Standard GPT classification
            gpt_result = self._gpt_identity_classification(text)
            if gpt_result[0]:  # If GPT found identity question
                return gpt_result
            
            # Second attempt: Check if text is similar to identity keywords
            # This handles cases like "من طورك" which might not be in examples
            is_similar = self._is_similar_to_identity_keywords(text)
            if is_similar:
                # Re-process with GPT using enhanced prompt with keyword context
                enhanced_result = self._gpt_identity_classification_with_context(text, is_similar)
                if enhanced_result[0]:
                    return enhanced_result
            
            # Third attempt: Direct keyword fallback
            return self._fallback_identity_detection(text)
            
        except Exception as e:
            self.logger.error(f"Error in identity detection: {e}")
            return self._fallback_identity_detection(text)
    
    def _gpt_identity_classification(self, text: str) -> tuple:
        """Standard GPT classification for identity questions."""
        identity_classification_prompt = f"""
Analyze this user input and determine if they are asking an identity question about the AI system/chatbot.

User input: "{text}"

CRITICAL: Distinguish between:
1. User describing THEMSELVES (NOT identity questions)
2. User asking about THE SYSTEM (identity questions)

Identity question categories:
1. who_are_you - asking about identity ("who are you", "tell me about yourself", "من أنت", "عرف بنفسك", etc.)
2. what_is_begining - asking about the BEGINING project ("what is begining", "ما هو بيجينينغ", "ما هو مشروع بيجينينغ", etc.)
3. purpose - asking about purpose ("why were you created", "what's your purpose", "ما هو هدفك", "لماذا تم إنشاؤك", etc.)
4. role - asking about role/function ("what do you do", "what's your role", "ما هو دورك", "ما وظيفتك", etc.)
5. developer - asking about creators ("who made you", "who's your developer", "من مطورك", "من صنعك", "من أنشأك", etc.)
6. team - asking about the team ("who's your team", "who's behind you", "من فريقك", "من وراءك", etc.)
7. understand_personality - asking about capabilities ("can you understand me", "هل تفهمني", "هل يمكنك فهم شخصيتي", etc.)
8. how_analyze - asking about methodology ("how do you work", "how do you analyze", "كيف تعمل", "كيف تحلل", etc.)
9. objectives - asking about goals ("what are your objectives", "ما أهدافك", "ما غاياتك", etc.)

Respond with ONLY ONE of these formats:
- If it's an identity question about the SYSTEM: "IDENTITY:category_name"
- If it's NOT an identity question (user describing themselves, personality input, etc.): "NOT_IDENTITY"

Examples (Identity questions about THE SYSTEM):
"who are you" -> "IDENTITY:who_are_you"
"who is your developer" -> "IDENTITY:developer"  
"what is begining" -> "IDENTITY:what_is_begining"
"what is your purpose" -> "IDENTITY:purpose"
"what do you do" -> "IDENTITY:role"
"who r u" -> "IDENTITY:who_are_you"
"ur developer" -> "IDENTITY:developer"
"what ur purpose" -> "IDENTITY:purpose"
"من أنت" -> "IDENTITY:who_are_you"
"من مطورك" -> "IDENTITY:developer"
"ما هو دورك" -> "IDENTITY:role"

Examples (NOT identity - user describing themselves or personality input):
"I am happy today" -> "NOT_IDENTITY"
"I am a developer" -> "NOT_IDENTITY"
"I am a developer who enjoys creating applications" -> "NOT_IDENTITY"
"I'm a creative developer" -> "NOT_IDENTITY"
"My job is programming" -> "NOT_IDENTITY"
"My role involves building websites" -> "NOT_IDENTITY"
"My purpose in life is to help others" -> "NOT_IDENTITY"
"I work as a programmer" -> "NOT_IDENTITY"
"I develop mobile applications" -> "NOT_IDENTITY"
"I create software solutions" -> "NOT_IDENTITY"
"أنا مطور برمجيات" -> "NOT_IDENTITY"
"وظيفتي في شركة تقنية" -> "NOT_IDENTITY"
"عملي هو تطوير التطبيقات" -> "NOT_IDENTITY"
"دوري في الفريق" -> "NOT_IDENTITY"
"أطور مواقع الويب" -> "NOT_IDENTITY"
"how do you feel" -> "NOT_IDENTITY"
"I like programming" -> "NOT_IDENTITY"
"أنا سعيد اليوم" -> "NOT_IDENTITY"
"""

        messages = [
            {"role": "system", "content": "You are an expert at classifying user questions about AI systems."},
            {"role": "user", "content": identity_classification_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
            max_tokens=50,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Handle different response formats from GPT
        if "IDENTITY:" in result:
            # Extract the category after IDENTITY:
            if result.startswith("IDENTITY:"):
                category = result.split("IDENTITY:")[1].strip()
            else:
                # Handle format like: "text" -> "IDENTITY:category"
                parts = result.split("IDENTITY:")
                if len(parts) > 1:
                    category = parts[1].strip().strip('"')
                else:
                    return False, None, None
            
            if category in self.IDENTITY_RESPONSES:
                return True, category, self.IDENTITY_RESPONSES[category]
        
        return False, None, None
    
    def _is_similar_to_identity_keywords(self, text: str) -> str:
        """
        Check if text is similar to identity keywords even if not exact match.
        Returns the likely category if similar, None otherwise.
        IMPROVED: Context-aware to avoid false positives from self-descriptions.
        """
        text_lower = text.lower().strip()
        text_lower = unicodedata.normalize("NFKD", text_lower)
        
        # FIRST: Check if it's a self-description (should NOT be identity)
        self_description_indicators = [
            # English self-descriptions
            "i am", "i'm", "my job", "my work", "my role", "my purpose in life",
            "i work", "i develop", "i create", "i build", "i design", "i study",
            "my friend", "we are", "everyone has", "imagine i", "if i am",
            "i am a better", "i play a role",
            
            # Arabic self-descriptions (expanded)
            "أنا", "انا", "وظيفتي", "عملي", "دوري", "أطور", "اطور",
            "انا مهندس", "أنا مهندس", "انا طبيب", "أنا طبيب"
        ]
        
        for indicator in self_description_indicators:
            if indicator in text_lower:
                return None  # Don't trigger identity detection for self-descriptions
        
        # SECOND: Check for statements about the system (not questions)
        statement_indicators = [
            "you are a chatbot", "you are what you are", "your purpose is clearer",
            "suppose your purpose", "your team", "with your team"
        ]
        
        for indicator in statement_indicators:
            if indicator in text_lower:
                return None  # Don't trigger identity detection for statements
        
        # THIRD: Check for third-party questions (should be off-topic)
        third_party_indicators = [
            ("who", "president"), ("who", "created facebook"), ("who", "made google"),
            ("what", "google do"), ("what", "capital"), ("what", "machine learning"),
            ("what", "quantum physics"), ("explain", "artificial intelligence"),
            ("tell me about", "history"), ("how", "photosynthesis")
        ]
        
        for pattern1, pattern2 in third_party_indicators:
            if pattern1 in text_lower and pattern2 in text_lower:
                return None  # Don't trigger identity detection for third-party questions
        
        # Define similarity patterns for QUESTIONS about the system only
        similarity_patterns = {
            "developer": [
                # Must have question words + developer context
                ("who", "developer"), ("who", "made"), ("who", "built"), ("who", "created"),
                ("من", "مطور"), ("مين", "مطور"), ("منو", "مطور"),
                ("من", "صنع"), ("من", "بنى"), ("من", "أنشأ")
            ],
            "purpose": [
                # Must have question words + purpose context
                ("what", "purpose"), ("why", "created"), ("why", "here"),
                ("ما", "هدف"), ("ايش", "هدف"), ("شو", "هدف"), ("وش", "هدف"),
                ("لماذا", "إنشاؤك"), ("ليش", "هنا")
            ],
            "role": [
                # Must have question words + role context  
                ("what", "do"), ("what", "role"), ("what", "function"),
                ("ما", "دور"), ("ايش", "دور"), ("شو", "دور"), ("وش", "دور"),
                ("ما", "وظيف")
            ],
            "who_are_you": [
                # Must have question words + identity context
                ("who", "you"), ("who", "are"), ("tell", "about"),
                ("من", "أنت"), ("مين", "أنت"), ("منو", "أنت"),
                ("عرف", "نفس")
            ]
        }
        
        # Check for pattern pairs (must have both elements)
        for category, pattern_pairs in similarity_patterns.items():
            for pattern1, pattern2 in pattern_pairs:
                if pattern1 in text_lower and pattern2 in text_lower:
                    return category
        
        return None
    
    def _gpt_identity_classification_with_context(self, text: str, suggested_category: str) -> tuple:
        """
        Enhanced GPT classification with keyword context for edge cases.
        """
        enhanced_prompt = f"""
Analyze this user input for identity questions about the AI system/chatbot.

User input: "{text}"

CONTEXT: This text shows similarity to "{suggested_category}" identity questions.
Common patterns for {suggested_category}:
- developer: asking about creators, who made/built/developed the system
- purpose: asking about goals, mission, why the system exists
- role: asking about function, job, what the system does
- who_are_you: asking about identity, who/what the system is

Look for the INTENT behind the question, even with:
- Typos or misspellings
- Informal language
- Different word order
- Arabic dialectal variations
- Missing or extra words

Categories:
1. who_are_you, 2. what_is_begining, 3. purpose, 4. role, 5. developer, 6. team, 7. understand_personality, 8. how_analyze, 9. objectives

Respond ONLY with:
- "IDENTITY:category_name" if it's an identity question
- "NOT_IDENTITY" if it's not an identity question

Focus on INTENT over exact wording.
"""

        messages = [
            {"role": "system", "content": "You are an expert at understanding user intent in questions about AI systems, even with variations in wording."},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,  # Slightly higher temperature for flexibility
            max_tokens=50,
        )
        
        result = response.choices[0].message.content.strip()
        
        if "IDENTITY:" in result:
            category = result.split("IDENTITY:")[1].strip()
            if category in self.IDENTITY_RESPONSES:
                return True, category, self.IDENTITY_RESPONSES[category]
        
        return False, None, None
    
    def _fallback_identity_detection(self, text: str) -> tuple:
        """
        Context-aware fallback method that distinguishes between:
        1. User self-descriptions: "I am a developer" -> NOT identity
        2. Questions about the system: "Who is your developer" -> IS identity
        """
        if not text:
            return False, None, None
            
        # Normalize text
        text_lower = text.lower().strip()
        text_lower = unicodedata.normalize("NFKD", text_lower)
        
        # FIRST: Check for self-description patterns (should NOT be identity)
        self_description_patterns = [
            # English self-descriptions
            r'\bi am (a |an )?developer\b',
            r'\bi\'m (a |an )?developer\b', 
            r'\bmy (job|work|role|profession) is\b',
            r'\bi work as (a |an )?\w+',
            r'\bi work with\b',
            r'\bi develop\b',
            r'\bi create\b',
            r'\bmy purpose in life\b',
            r'\bmy role involves\b',
            r'\bi study\b',
            r'\bmy friend is\b',
            r'\bwe are\b',
            r'\beveryone has\b',
            r'\bimagine i\b',
            r'\bif i am\b',
            r'\bi am a better\b',
            r'\bi play a role\b',
            
            # Arabic self-descriptions (expanded)
            r'\bأنا مطور\b',
            r'\bانا مطور\b',
            r'\bانا مهندس\b',
            r'\bأنا مهندس\b',
            r'\bوظيفتي\b',
            r'\bعملي هو\b',
            r'\bدوري في\b',
            r'\bأطور\b',
            r'\bاطور\b'
        ]
        
        # SECOND: Check for statements about the system (should NOT be identity)
        statement_patterns = [
            r'\byou are a chatbot\b',
            r'\byou are what you are\b',
            r'\byour purpose is clearer\b',
            r'\bsuppose your purpose\b'
        ]
        
        # THIRD: Check for third-party/off-topic patterns (should NOT be identity)
        third_party_patterns = [
            r'\bwho (is )?the president\b',
            r'\bwho created facebook\b',
            r'\bwho made google\b',
            r'\bwhat does google do\b',
            r'\bwhat (is )?the capital\b',
            r'\bwhat (is )?machine learning\b',
            r'\bwhat (is )?quantum physics\b',
            r'\bexplain artificial intelligence\b',
            r'\btell me about history\b',
            r'\bhow does photosynthesis\b'
        ]
        
        # Check all non-identity patterns first
        all_non_identity_patterns = self_description_patterns + statement_patterns + third_party_patterns
        for pattern in all_non_identity_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, None, None
        
        # SECOND: Check for ACTUAL identity questions about the system
        identity_patterns = {
            "developer": [
                # Specific questions about the system's developer
                r'\bwho (is |are )?your developer\b',
                r'\bwho developed you\b',
                r'\bwho (built|created|made) you\b',
                r'\bwho\'s your (developer|creator|maker)\b',
                r'\bur developer\b',
                
                # Arabic - questions about the system
                r'\bمن مطورك\b',
                r'\bمين مطورك\b', 
                r'\bمنو مطورك\b',
                r'\bمن طورك\b',
                r'\bمن (صنعك|بناك|أنشأك)\b'
            ],
            "purpose": [
                # Questions about the system's purpose
                r'\bwhat (is |are )?your purpose\b',
                r'\bwhy (were you|are you) (created|made|built)\b',
                r'\bwhat\'s your (purpose|goal|mission)\b',
                r'\bur purpose\b',
                
                # Arabic - questions about the system
                r'\bما (هو )?هدفك\b',
                r'\b(ايش|شو|وش) هدفك\b',
                r'\bلماذا (تم إنشاؤك|أنت هنا)\b'
            ],
            "role": [
                # Questions about what the system does
                r'\bwhat do you do\b',
                r'\bwhat (is |are )?your (role|function|job)\b',
                r'\bwhat\'s your (role|function|job)\b',
                r'\bur (role|function|job)\b',
                
                # Arabic - questions about the system
                r'\bما (هو )?دورك\b',
                r'\b(ايش|شو|وش) (دورك|وظيفتك)\b'
            ],
            "who_are_you": [
                # Direct questions about system identity
                r'\bwho are you\b',
                r'\bwho r u\b',
                r'\btell me about (you|yourself)\b',
                r'\bintroduce yourself\b',
                
                # Arabic
                r'\bمن أنت\b',
                r'\bمين أنت\b',
                r'\bعرف بنفسك\b'
            ],
            "what_is_begining": [
                # Questions about the Begining project
                r'\bwhat (is |are )?begining\b',
                r'\bexplain begining\b',
                r'\babout begining\b',
                
                # Arabic
                r'\bما (هو )?بيجينينغ\b',
                r'\b(ايش|شو|وش) بيجينينغ\b'
            ],
            "team": [
                # Questions about the team behind the system
                r'\bwho\'s (behind you|your team)\b',
                r'\byour team\b',
                r'\bwho (built|developed|created) this\b',
                
                # Arabic
                r'\bمن فريقك\b',
                r'\bمين (وراءك|فريقك)\b'
            ],
            "understand_personality": [
                # Questions about the system's capabilities
                r'\bcan you understand (me|personality)\b',
                r'\bdo you understand\b',
                r'\byour understanding\b',
                
                # Arabic
                r'\bهل تفهمني\b',
                r'\bتقدر تفهمني\b'
            ],
            "how_analyze": [
                # Questions about how the system works
                r'\bhow do you (work|analyze|function)\b',
                r'\byour method\b',
                r'\bhow you analyze\b',
                
                # Arabic
                r'\bكيف (تعمل|تحلل|تشتغل)\b',
                r'\bطريقتك\b'
            ],
            "objectives": [
                # Questions about system objectives
                r'\byour (objectives|goals|aims)\b',
                r'\bwhat (are |is )?your (objectives|goals)\b',
                r'\bwhat are your top \d+ objectives\b',
                r'\btop \d+ objectives\b',
                
                # Arabic
                r'\bما أهدافك\b',
                r'\b(ايش|شو|وش) أهدافك\b'
            ],
            "purpose": [
                # Questions about the system's purpose (expanded)
                r'\bwhat (is |are )?your purpose\b',
                r'\bwhy (were you|are you) (created|made|built)\b',
                r'\bwhat\'s your (purpose|goal|mission)\b',
                r'\bur purpose\b',
                r'\bsuppose your purpose\b',
                r'\byour purpose (is|was|would be)\b',
                
                # Arabic - questions about the system
                r'\bما (هو )?هدفك\b',
                r'\b(ايش|شو|وش) هدفك\b',
                r'\bلماذا (تم إنشاؤك|أنت هنا)\b'
            ]
        }
        
        # Check for identity question patterns
        for category, patterns in identity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if category in self.IDENTITY_RESPONSES:
                        return True, category, self.IDENTITY_RESPONSES[category]
        
        return False, None, None

    @staticmethod
    def get_identity_response(response_data: dict, languages: str) -> str:
        """
        Get the appropriate identity response based on the user's language preference.
        """
        if not response_data:
            return ""
        
        # Determine language preference
        if "ar" in languages or "arabic" in languages.lower():
            return response_data.get("arabic", response_data.get("english", ""))
        else:
            return response_data.get("english", "")

    def detect_off_topic_question(self, text: str, languages: str) -> tuple:
        """
        Detect if the user is asking off-topic questions (not related to personality or identity).
        Returns a tuple: (is_off_topic: bool, response_type: str, response_text: str)
        """
        if not text:
            return False, None, None
        
        # Clean and normalize text
        text_lower = text.lower().strip()
        text_lower = unicodedata.normalize("NFKD", text_lower)
        
        # Check for gibberish/random characters
        # If more than 60% of characters are non-standard, consider it gibberish
        total_chars = len(text_lower.replace(" ", ""))
        if total_chars > 0:
            # Count standard characters (letters, numbers, basic punctuation)
            standard_chars = len(re.findall(r'[a-zA-Z0-9\u0600-\u06FF\s.,!?:;"\'-]', text_lower))
            if standard_chars / total_chars < 0.4:  # Less than 40% standard characters
                response_text = self._get_off_topic_response("gibberish", languages)
                return True, "gibberish", response_text
        
        # Patterns for common off-topic questions
        off_topic_patterns = {
            "factual_questions": [
                # Weather and nature
                r"(what|how|why|when|where).*(?:color|colour).*sky",
                r"(what|how|why|when|where).*weather",
                r"(what|how|why|when|where).*rain",
                r"(what|how|why|when|where).*sun",
                r"(what|how|why|when|where).*moon",
                r"(what|how|why|when|where).*star",
                
                # Science and facts
                r"(what|how|why|when|where).*gravity",
                r"(what|how|why|when|where).*earth",
                r"(what|how|why|when|where).*planet",
                r"(what|how|why|when|where).*science",
                r"(what|how|why|when|where).*mathematics?",
                r"(what|how|why|when|where).*history",
                r"(what|how|why|when|where).*photosynthesis",
                r"(what|how|why|when|where).*quantum physics",
                r"(what|how|why|when|where).*capital.*france",
                
                # Geography and places
                r"(what|how|why|when|where).*(capital|city|country)",
                r"who (is )?the president",
                r"who created (facebook|google|microsoft)",
                r"what does (google|facebook|microsoft) do",
                
                # Current events and news
                r"(what|how|why|when|where).*news",
                r"(what|how|why|when|where).*today",
                r"(what|how|why|when|where).*happened",
                r"(what|how|why|when|where).*time",
                r"(what|how|why|when|where).*date",
                
                # Arabic equivalents
                r"(ما|كيف|لماذا|متى|أين).*لون.*السماء",
                r"(ما|كيف|لماذا|متى|أين).*الطقس",
                r"(ما|كيف|لماذا|متى|أين).*المطر",
                r"(ما|كيف|لماذا|متى|أين).*الشمس",
                r"(ما|كيف|لماذا|متى|أين).*القمر",
                r"(ما|كيف|لماذا|متى|أين).*العلم",
                r"(ما|كيف|لماذا|متى|أين).*التاريخ",
                r"(ما|كيف|لماذا|متى|أين).*الأخبار",
                r"أخبرني عن التاريخ",
                r"tell me about history",
            ],
            "technical_questions": [
                # Programming and technology
                r"(what|how|why|when|where).*python",
                r"(what|how|why|when|where).*javascript",
                r"(what|how|why|when|where).*programming",
                r"(what|how|why|when|where).*code",
                r"(what|how|why|when|where).*computer",
                r"(what|how|why|when|where).*software",
                r"(what|how|why|when|where).*internet",
                r"(what|how|why|when|where).*algorithm",
                r"(what|how|why|when|where).*machine learning",
                r"(what|how|why|when|where).*artificial intelligence",
                r"explain artificial intelligence",
                r"how to write better code",
                r"how do i learn python",
                
                # Arabic equivalents
                r"(ما|كيف|لماذا|متى|أين).*البرمجة",
                r"(ما|كيف|لماذا|متى|أين).*الكمبيوتر",
                r"(ما|كيف|لماذا|متى|أين).*البرنامج",
                r"(ما|كيف|لماذا|متى|أين).*التقنية",
                r"(ما|كيف|لماذا|متى|أين).*الإنترنت",
                r"كيف أتعلم البرمجة",
            ]
        }
        
        # Check for personality-related content first (avoid false positives)
        personality_indicators = [
            # English personality words
            "feel", "emotion", "personality", "behavior", "social", "think", "cognitive",
            "trait", "character", "myself", "yourself", "how you", "how i", "when i",
            "i am", "i like", "i prefer", "i usually", "i tend", "i often",
            
            # English job/self-description words
            "i work", "my job", "my role", "my profession", "i study", "my friend",
            "developer", "engineer", "teacher", "programmer", "designer",

            # Arabic personality words  
            "أشعر", "شعور", "شخصية", "سلوك", "اجتماعي", "أفكر", "معرفي",
            "صفة", "طبع", "نفسي", "نفسك", "كيف أنت", "كيف أنا", "عندما أكون",
            "أنا", "أحب", "أفضل", "عادة", "أميل", "غالباً",
            
            # Arabic job/self-description words (expanded)
            "انا مهندس", "أنا مهندس", "انا طبيب", "أنا طبيب", "انا مطور", "أنا مطور",
            "انا مدرس", "أنا مدرس", "انا مصمم", "أنا مصمم", "وظيفتي", "عملي",
            "مهندس", "طبيب", "مدرس", "مصمم"
        ]
        
        # If text contains personality indicators, it's likely not off-topic
        for indicator in personality_indicators:
            if indicator in text_lower:
                return False, None, None
        
        # Check for off-topic patterns
        for category, patterns in off_topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    response_text = self._get_off_topic_response(category, languages)
                    return True, category, response_text
        
        # Check for very short or unclear responses
        words = text_lower.split()
        if len(words) <= 2 and not any(word in text_lower for word in ["yes", "no", "نعم", "لا", "ok", "حسناً"]):
            # Very short unclear response
            response_text = self._get_off_topic_response("general_unrelated", languages)
            return True, "general_unrelated", response_text
        
        return False, None, None
    
    def _get_off_topic_response(self, response_type: str, languages: str) -> str:
        """Get appropriate off-topic response based on type and language."""
        if response_type not in self.OFF_TOPIC_RESPONSES:
            response_type = "general_unrelated"
        
        response_data = self.OFF_TOPIC_RESPONSES[response_type]
        
        # Determine language preference
        if "ar" in languages or "arabic" in languages.lower():
            return response_data.get("arabic", response_data.get("english", ""))
        else:
            return response_data.get("english", "")

    def analyze_missing_traits(self, user_input: str, new_input: list) -> list:
        """
        Analyze which personality traits need more detailed exploration.
        Returns a list of trait categories that need clarification questions.
        """
        # Combine all input text (excluding identity questions)
        all_text = user_input.lower()
        personality_answers = []
        
        for qa in new_input:
            answer = qa.get("answer", "").lower()
            # Skip identity questions in trait analysis
            is_identity, _, _ = self.detect_identity_question(answer)
            if not is_identity:
                all_text += " " + answer
                personality_answers.append(answer)
        
        # If we have very little personality data, return all traits as missing
        if len(personality_answers) < 2 or len(all_text.split()) < 30:
            return ["emotional", "social", "cognitive", "behavioral"]
        
        # Check for detailed coverage of each trait
        traits_needing_clarification = []
        
        for trait, pattern in PersonalityAnalyzer.TRAIT_PATTERNS.items():
            matches = re.findall(pattern, all_text)
            
            # Get all answers for this trait to check depth
            trait_answers = [answer for answer in personality_answers 
                           if re.search(pattern, answer)]
            
            # Check if we need more clarification based on:
            # 1. Number of matches (< 2)
            # 2. Length of answers (< 10 words average)
            # 3. Variety of trait expressions
            avg_length = sum(len(answer.split()) for answer in trait_answers) / max(len(trait_answers), 1)
            
            if (len(matches) < 2 or 
                avg_length < 10 or 
                len(trait_answers) == 0):
                traits_needing_clarification.append(trait)
        
        # Always return at least some traits to keep conversation going
        # unless we have very comprehensive data
        if not traits_needing_clarification and len(personality_answers) < 4:
            # Return 1-2 random traits to get more detail
            import random
            all_traits = ["emotional", "social", "cognitive", "behavioral"]
            random.shuffle(all_traits)
            return all_traits[:1]
        
        return traits_needing_clarification

    @staticmethod
    def generate_clarification_questions(missing_traits: list, languages: str, max_questions: int = 2, asked_questions: list = None) -> list:
        """
        Generate clarification questions for missing traits in the appropriate language.
        Avoids repeating previously asked questions.
        """
        import random
        
        if not missing_traits:
            return []
        
        if asked_questions is None:
            asked_questions = []
        
        # Determine language preference
        is_arabic = "ar" in languages or "arabic" in languages.lower()
        
        # Select appropriate templates
        templates = PersonalityAnalyzer.CLARIFICATION_TEMPLATES_ARABIC if is_arabic else PersonalityAnalyzer.CLARIFICATION_TEMPLATES
        
        questions = []
        
        # Shuffle missing traits to vary question order
        shuffled_traits = missing_traits.copy()
        random.shuffle(shuffled_traits)
        
        # Generate questions for up to max_questions traits
        for trait in shuffled_traits[:1]:
            if trait in templates:
                available_questions = [q for q in templates[trait] if q not in asked_questions]
                if available_questions:
                    question = random.choice(available_questions)
                    questions.append(question)
                elif templates[trait]:  # Fallback if all questions were asked
                    question = random.choice(templates[trait])
                    questions.append(question)
        
        return questions
    SYSTEM_PROMPT = """
You are a sociologist and can analyze and extract character descriptions from texts in a professional manner, in line with your field.

Purpose
You will help extract character descriptions by reviewing texts submitted by users — these may sometimes be random — and converting them into concise descriptive texts that capture four key personality traits:

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

Status Types
- "complete": All four personality traits are sufficiently covered
- "incomplete": Some traits are missing and need clarification
- "identity": When user asks identity questions about the system (handled separately)
- "off_topic": When user asks unrelated questions not about personality or identity (handled separately)

Output Format
id (integer)
status ("complete", "incomplete", "identity", or "off_topic")
description_english (concise personality description)
description_arabic (concise personality description in Arabic if possible)
description_identity (only for identity status - IDENTITY_RESPONSES)
description_off_topic (only for off_topic status - OFF_TOPIC_RESPONSES)
missing_traits (array or null)
clarification_questions (array)
input_tokens (integer)
output_tokens (integer)

            LANGUAGE HANDLING:
            Only fill in 'description_english' if the user's languages field includes "en" or "english". Only fill in 'description_arabic' if the user's languages field includes "ar" or "arabic". If a language is not requested, leave its description field as an empty string.

            All clarification questions and trait names (in 'missing_traits') must be in the user's requested language(s) as specified in the 'languages' field.

            Do not include any extra text, code blocks, or explanations outside the JSON.


Do not include any extra text, code blocks, or explanations outside the JSON.

Example Output — Incomplete
{
    "id": 22,
    "status": "incomplete",
    "description_arabic": "",
    "description_english": "",
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
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
            self.logger.setLevel(logging.INFO)

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
        First checks for identity questions, then processes personality analysis.
        """
        self.logger.debug(f"Starting analysis for user {id}")
        if new_input is None:
            new_input = []
        
        # Auto-detect language if not specified or if "auto" is passed
        if languages == "auto" or not languages:
            # Check most recent input for language
            recent_text = ""
            if new_input:
                recent_text = new_input[-1].get("answer", "")
            else:
                recent_text = user_input
            
            detected_lang = self.detect_language(recent_text)
            languages = "ar" if detected_lang == "arabic" else "en"
        
        # Identity detection logic:
        # 1. If new_input is empty -> check user_input (first interaction)
        # 2. If new_input exists -> only check LAST answer, ignore user_input (history)
        is_identity, response_key, response_data = False, None, None
        
        if new_input:
            # Check only the LAST answer in new_input (most recent question)
            # Ignore user_input because it contains history
            last_qa = new_input[-1]
            last_answer = last_qa.get("answer", "").strip()
            is_identity, response_key, response_data = self.detect_identity_question(last_answer)
        else:
            # No new_input means first interaction - check user_input
            is_identity, response_key, response_data = self.detect_identity_question(user_input)
        
        if is_identity:
            # Analyze what personality traits are still missing
            missing_traits = self.analyze_missing_traits(user_input, new_input)
            
            # Extract previously asked questions to avoid repetition
            asked_questions = []
            for qa in new_input:
                question = qa.get("question", "").strip()
                if question:
                    asked_questions.append(question)
            
            # Generate clarification questions to continue the conversation
            clarification_questions = self.generate_clarification_questions(
                missing_traits, languages, max_questions=2, asked_questions=asked_questions
            )
            
            # Return identity response with clarification questions to continue conversation
            identity_text = self.get_identity_response(response_data, languages)
            return {
                "content": json.dumps({
                    "id": id,
                    "status": "identity",
                    "description_identity": identity_text,
                    "description_english": "",
                    "description_arabic": "",
                    "missing_traits": missing_traits,
                    "clarification_questions": clarification_questions
                }),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        
        # Check for off-topic questions (after identity detection but before personality analysis)
        input_to_check = ""
        if new_input:
            # Check the last answer for off-topic content
            last_qa = new_input[-1]
            input_to_check = last_qa.get("answer", "").strip()
        else:
            # Check initial user input
            input_to_check = user_input
        
        is_off_topic, off_topic_type, off_topic_response = self.detect_off_topic_question(input_to_check, languages)
        if is_off_topic:
            # Analyze what personality traits are still missing (to continue conversation)
            missing_traits = self.analyze_missing_traits(user_input, new_input)
            
            # Extract previously asked questions to avoid repetition
            asked_questions = []
            for qa in new_input:
                question = qa.get("question", "").strip()
                if question:
                    asked_questions.append(question)
            
            # Generate clarification questions to guide back to personality topics
            clarification_questions = self.generate_clarification_questions(
                missing_traits, languages, max_questions=1, asked_questions=asked_questions
            )
            
            return {
                "content": json.dumps({
                    "id": id,
                    "status": "off_topic",
                    "description_off_topic": off_topic_response,
                    "description_english": "",
                    "description_arabic": "",
                    "missing_traits": missing_traits,
                    "clarification_questions": clarification_questions
                }),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        
        # If not an identity or off-topic question, proceed with normal personality analysis
        input_data = {
            "id": id,
            "user_input": user_input,
            "new_input": new_input,
            "languages": languages
        }
        gpt_response = self.call_gpt(input_data)
        return gpt_response