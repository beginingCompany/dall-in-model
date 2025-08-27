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
        Use GPT to intelligently detect if the user is asking an identity question about the system.
        Returns a tuple: (is_identity_question: bool, response_key: str, response_data: dict)
        """
        if not text:
            return False, None, None
            
        # Use GPT to classify the question
        identity_classification_prompt = f"""
Analyze this user input and determine if they are asking an identity question about the AI system/chatbot.

User input: "{text}"

NOTE: YOU MAY SEE SOME MISTAKES IN THE EXAMPLES BELOW, PLEASE FOLLOW THE INTENT RATHER THAN THE EXACT TEXT.
IMPORTANT: Pay attention to informal variations and common typos. AND USERS MAY MISTYPE THEIR QUESTIONS.
Identity question categories:
1. who_are_you - asking about identity ("who are you", "tell me about yourself", "من أنت", "عرف بنفسك", etc.)
2. what_is_begining - asking about the BEGINING project ("what is begining", "ما هو بيجينينغ", "ما هو مشروع بيجينينغ", etc.)
3. purpose - asking about purpose ("why were you created", "what's your purpose", "ما هو هدفك", "لماذا تم إنشاؤك", etc.)
4. role - asking about role/function ("what do you do", "what's your role", "ما هو دورك", "ما وظيفتك", etc.)
5. developer - asking about creators ("who made you", "who's your developer", "من مطورك","من طورك", "من صنعك", "من أنشأك", etc.)
6. team - asking about the team ("who's your team", "who's behind you", "من فريقك", "من وراءك", etc.)
7. understand_personality - asking about capabilities ("can you understand me", "هل تفهمني", "هل يمكنك فهم شخصيتي", etc.)
8. how_analyze - asking about methodology ("how do you work", "how do you analyze", "كيف تعمل", "كيف تحلل", etc.)
9. objectives - asking about goals ("what are your objectives", "ما أهدافك", "ما غاياتك", etc.)

Respond with ONLY ONE of these formats:
- If it's an identity question: "IDENTITY:category_name"
- If it's NOT an identity question: "NOT_IDENTITY"

Examples (English - Formal):
"who are you" -> "IDENTITY:who_are_you"
"who is your developer" -> "IDENTITY:developer"  
"what is begining" -> "IDENTITY:what_is_begining"
"what is your purpose" -> "IDENTITY:purpose"
"what do you do" -> "IDENTITY:role"

Examples (English - Informal):
"who r u" -> "IDENTITY:who_are_you"
"who u" -> "IDENTITY:who_are_you"
"ur identity" -> "IDENTITY:who_are_you"
"who ur developer" -> "IDENTITY:developer"
"ur developer" -> "IDENTITY:developer"
"ur creator" -> "IDENTITY:developer"
"what ur purpose" -> "IDENTITY:purpose"
"ur purpose" -> "IDENTITY:purpose"
"why u here" -> "IDENTITY:purpose"
"ur goal" -> "IDENTITY:purpose"
"ur mission" -> "IDENTITY:purpose"
"what u do" -> "IDENTITY:role"
"ur job" -> "IDENTITY:role"
"ur role" -> "IDENTITY:role"
"ur team" -> "IDENTITY:team"

Examples (Arabic):
"من أنت" -> "IDENTITY:who_are_you"
"من مطورك" -> "IDENTITY:developer"
"مطورك" -> "IDENTITY:developer"
"ما هو مشروع بيجينينغ" -> "IDENTITY:what_is_begining"
"ما هو دورك" -> "IDENTITY:role"
"دورك" -> "IDENTITY:role"
"هدفك" -> "IDENTITY:purpose"

Examples (Non-identity):
"I am happy today" -> "NOT_IDENTITY"
"how do you feel" -> "NOT_IDENTITY"
"I like programming" -> "NOT_IDENTITY"
"أنا سعيد اليوم" -> "NOT_IDENTITY"
"أنا سعيد اليوم" -> "NOT_IDENTITY"
"""

        try:
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
            
        except Exception as e:
            self.logger.error(f"Error in identity detection: {e}")
            # Fallback to simple keyword matching if GPT fails
            return self._fallback_identity_detection(text)
    
    def _fallback_identity_detection(self, text: str) -> tuple:
        """
        Enhanced fallback method using flexible keyword matching if GPT detection fails.
        Handles variations like 'ur purpose', 'what ur role', etc.
        """
        if not text:
            return False, None, None
            
        text_lower = text.lower().strip()
        
        # Enhanced keyword-based fallback with more variations and flexibility
        identity_keywords = {
            "developer": [
                "developer", "made you", "built you", "created you", "creator", "ur developer", 
                "your developer", "who developed", "who built", "who created", "ur creator",
                "your creator", "who made", "developed by", "created by", "who ur developer",
                "من مطورك", "من صنعك", "من أنشأك", "من بناك", "مطور", "مطورك", "من صممك"
            ],
            "purpose": [
                "purpose", "why were you created", "why are you here", "ur purpose", "your purpose",
                "what ur purpose", "what is ur purpose", "what's ur purpose", "why u here",
                "why you here", "what for", "ur goal", "your goal", "ur mission", "your mission",
                "ما هو هدفك", "لماذا تم إنشاؤك", "هدفك", "غايتك", "مهمتك", "لماذا أنت هنا"
            ],
            "role": [
                "what do you do", "your role", "your function", "ur role", "ur function",
                "what ur role", "what is ur role", "what's ur role", "what u do", "ur job",
                "your job", "ur work", "your work", "ur task", "your task",
                "ما هو دورك", "ما وظيفتك", "دورك", "وظيفتك", "عملك", "مهامك"
            ],
            "who_are_you": [
                "who are you", "who r u", "who ru", "who u", "tell me about you", "introduce yourself",
                "about you", "who is this", "ur identity", "your identity",
                "من أنت", "عرف بنفسك", "من انت", "هويتك"
            ],
            "what_is_begining": [
                "what is begining", "begining", "explain begining", "about begining",
                "begining project", "what begining", "tell me about begining",
                "ما هو بيجينينغ", "ما هو مشروع بيجينينغ", "بيجينينغ", "مشروع بيجينينغ"
            ],
            "team": [
                "your team", "who's behind you", "who's working with you", "ur team",
                "who behind you", "ur colleagues", "your colleagues", "who with you",
                "من فريقك", "من وراءك", "فريقك", "زملاؤك", "من معك"
            ],
            "understand_personality": [
                "can you understand", "do you understand", "understand me", "analyze me",
                "can u understand", "do u understand", "ur understanding", "your understanding",
                "هل تفهمني", "هل يمكنك فهمي", "تفهمني", "تحليلي"
            ],
            "how_analyze": [
                "how do you work", "how do you analyze", "how u work", "how u analyze",
                "ur method", "your method", "how you function", "how u function",
                "كيف تعمل", "كيف تحلل", "طريقتك", "كيف تشتغل"
            ],
            "objectives": [
                "your objectives", "ur objectives", "your goals", "ur goals", "objectives",
                "what ur objectives", "what are ur objectives", "ur aims", "your aims",
                "ما أهدافك", "ما غاياتك", "أهدافك", "غاياتك"
            ]
        }
        
        # Use flexible matching - check if any keyword appears in the text
        for category, keywords in identity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category in self.IDENTITY_RESPONSES:
                        return True, category, self.IDENTITY_RESPONSES[category]
        
        # Additional pattern-based matching for even more flexibility
        # Handle patterns like "what's ur [X]", "ur [X]", etc.
        flexible_patterns = {
            "purpose": ["purpose", "goal", "mission", "هدف", "غاية", "مهمة"],
            "role": ["role", "job", "work", "function", "task", "دور", "وظيفة", "عمل", "مهمة"],
            "developer": ["developer", "creator", "maker", "مطور", "صانع", "منشئ"],
            "team": ["team", "colleagues", "group", "فريق", "زملاء", "مجموعة"]
        }
        
        for category, pattern_words in flexible_patterns.items():
            for word in pattern_words:
                # Check patterns like "ur [word]", "your [word]", "what's ur [word]", etc.
                if (f"ur {word}" in text_lower or f"your {word}" in text_lower or 
                    f"what ur {word}" in text_lower or f"what's ur {word}" in text_lower or
                    f"what is ur {word}" in text_lower or f"what's your {word}" in text_lower):
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
            # Need multiple matches or detailed responses for each trait
            if len(matches) < 2:
                traits_needing_clarification.append(trait)
        
        # Always return at least some traits to keep conversation going
        # unless we have very comprehensive data
        if not traits_needing_clarification and len(personality_answers) < 4:
            # Return 1-2 random traits to get more detail
            import random
            all_traits = ["emotional", "social", "cognitive", "behavioral"]
            random.shuffle(all_traits)
            return all_traits[:2]
        
        return traits_needing_clarification

    @staticmethod
    def generate_clarification_questions(missing_traits: list, languages: str, max_questions: int = 2) -> list:
        """
        Generate clarification questions for missing traits in the appropriate language.
        """
        import random
        
        if not missing_traits:
            return []
        
        # Determine language preference
        is_arabic = "ar" in languages or "arabic" in languages.lower()
        
        # Select appropriate templates
        templates = PersonalityAnalyzer.CLARIFICATION_TEMPLATES_ARABIC if is_arabic else PersonalityAnalyzer.CLARIFICATION_TEMPLATES
        
        questions = []
        
        # Shuffle missing traits to vary question order
        shuffled_traits = missing_traits.copy()
        random.shuffle(shuffled_traits)
        
        # Generate questions for up to max_questions traits
        for trait in shuffled_traits[:max_questions]:
            if trait in templates:
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

Output Format
id (integer)
status ("complete", "incomplete", or "identity")
description_english (concise personality description)
description_arabic (concise personality description in Arabic if possible)
description_identity (only for identity status - IDENTITY_RESPONSES)
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
        First checks for identity questions, then processes personality analysis.
        """
        self.logger.debug(f"Starting analysis for user {id}")
        if new_input is None:
            new_input = []
        
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
            
            # Generate clarification questions to continue the conversation
            clarification_questions = self.generate_clarification_questions(missing_traits, languages, max_questions=2)
            
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
        
        # If not an identity question, proceed with normal personality analysis
        input_data = {
            "id": id,
            "user_input": user_input,
            "new_input": new_input,
            "languages": languages
        }
        gpt_response = self.call_gpt(input_data)
        return gpt_response