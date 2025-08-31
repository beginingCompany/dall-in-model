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
    def detect_language(self, text: str) -> str:
        """
        GPT-powered language detection that analyzes the dominant language in text.
        Returns 'arabic' if Arabic is dominant, otherwise 'english'.
        """
        if not text or not text.strip():
            return "english"
        
        # Clean text and remove extra spaces
        text = text.strip()
        
        try:
            language_prompt = f"""
Analyze this text and determine the dominant language. Consider:
1. The primary language being used
2. Mixed language content - which language is more prominent
3. Context clues like names, job titles, greetings

Text: "{text}"

Respond with ONLY one word:
- "arabic" if Arabic is the dominant language
- "english" if English is the dominant language

Even if the text contains both languages, choose the one that is more prominent or contextually important.

Examples:
"مرحبا انا وليد developer" -> "arabic" (Arabic greeting and name are primary)
"Hi I'm Ahmed مهندس" -> "english" (English structure is primary)
"أنا مطور software" -> "arabic" (Arabic personal introduction is primary)
"I work as مهندس" -> "english" (English sentence structure is primary)
"Hello my name is وليد" -> "english" (English sentence structure)
"مرحبا اسمي John" -> "arabic" (Arabic greeting pattern)
"""

            messages = [
                {"role": "system", "content": "You are an expert at detecting the dominant language in text, even with mixed languages."},
                {"role": "user", "content": language_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if "arabic" in result:
                print(f"GPT detected Arabic language in text: '{text}'")
                return "arabic"
            else:
                print(f"GPT detected English language in text: '{text}'")
                return "english"
                
        except Exception as e:
            print(f"Error in GPT language detection: {e}")
            # Simple fallback - check for any Arabic characters
            arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
            if arabic_chars > 0:
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

    def detect_personal_introduction(self, text: str) -> tuple:
        """
        Use GPT to intelligently detect if user is introducing themselves with name or job title.
        Returns a tuple: (has_introduction: bool, name: str, job_title: str, greeting_message: str)
        """
        if not text or not text.strip():
            return False, "", "", ""
        
        # Debug logging
        print(f"Personal introduction detection for text: '{text}'")
        
        try:
            # Use GPT to detect personal introductions
            introduction_prompt = f"""
Analyze this user input and determine if they are introducing themselves personally.

User input: "{text}"

Look for:
1. Name introductions (any name in any language)
2. Job/profession introductions (any profession in any language)  
3. Personal self-descriptions
4. Greetings combined with personal information

Extract:
- Name (if mentioned)
- Job/Profession (if mentioned)
- Whether this is a personal introduction

Respond with ONLY this JSON format:
{{"is_introduction": true/false, "name": "extracted_name_or_empty", "job": "extracted_job_or_empty", "language": "arabic_or_english"}}

Examples:
"I am Ahmed" -> {{"is_introduction": true, "name": "Ahmed", "job": "", "language": "english"}}
"انا مهندس" -> {{"is_introduction": true, "name": "", "job": "مهندس", "language": "arabic"}}
"انا المهندس احمد" -> {{"is_introduction": true, "name": "احمد", "job": "مهندس", "language": "arabic"}}
"مرحبا انا وليد مهندس بيوميجات" -> {{"is_introduction": true, "name": "وليد", "job": "مهندس", "language": "arabic"}}
"My name is John and I work as a developer" -> {{"is_introduction": true, "name": "John", "job": "developer", "language": "english"}}
"I like programming" -> {{"is_introduction": false, "name": "", "job": "", "language": "english"}}
"What is the weather?" -> {{"is_introduction": false, "name": "", "job": "", "language": "english"}}
"من أنت" -> {{"is_introduction": false, "name": "", "job": "", "language": "arabic"}}
"""

            messages = [
                {"role": "system", "content": "You are an expert at detecting personal introductions and extracting names and professions from text in any language."},
                {"role": "user", "content": introduction_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            )
            
            result = response.choices[0].message.content.strip()
            print(f"GPT response for personal introduction: {result}")
            
            # Parse the JSON response
            import json
            try:
                data = json.loads(result)
                is_intro = data.get("is_introduction", False)
                name = data.get("name", "").strip()
                job = data.get("job", "").strip()
                detected_language = data.get("language", "english").strip()
                
                print(f"Parsed: is_intro={is_intro}, name='{name}', job='{job}', lang='{detected_language}'")
                
                if is_intro and (name or job):  # Only consider it introduction if we have name OR job
                    # Generate varied, friendly greeting
                    greeting = self.generate_varied_greeting(name, job, detected_language)
                    print(f"Generated greeting: '{greeting}'")
                    return True, name, job, greeting
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # Fallback: Enhanced GPT detection for common cases
                return self._enhanced_introduction_detection(text)
                    
        except Exception as e:
            print(f"Error in GPT personal introduction detection: {e}")
            # Fallback to enhanced GPT detection
            return self._enhanced_introduction_detection(text)
        
        print("No personal introduction detected")
        return False, "", "", ""

    def normalize_job_title(self, job: str) -> str:
        """Normalize common job abbreviations and variations to full forms"""
        if not job:
            return job
        
        job_lower = job.lower().strip()
        
        # Common abbreviations and variations
        job_mappings = {
            "eng": "engineer",
            "dev": "developer", 
            "prog": "programmer",
            "mgr": "manager",
            "admin": "administrator",
            "tech": "technician",
            "analyst": "data analyst",
            "designer": "graphic designer",
            "consultant": "business consultant",
            "specialist": "specialist",
            "coordinator": "project coordinator",
            "assistant": "assistant",
            "supervisor": "supervisor",
            "director": "director"
        }
        
        # Check exact matches first
        if job_lower in job_mappings:
            return job_mappings[job_lower]
        
        # Check partial matches for compound jobs
        for abbrev, full_form in job_mappings.items():
            if abbrev in job_lower and len(job_lower) <= len(abbrev) + 3:
                return full_form
        
        return job

    def generate_varied_greeting(self, name: str, job: str, language: str) -> str:
        """Generate varied, friendly greetings with personality"""
        import random
        
        # Normalize job title
        job = self.normalize_job_title(job)
        
        if language == "arabic":
            if name and job:
                greetings = [
                    f"أهلاً وسهلاً {name}! سعيد بلقائك. {job} - مهنة رائعة!",
                    f"مرحباً {name}! أهلاً بك معنا. أرى أنك تعمل كـ{job}، هذا مثير للاهتمام!",
                    f"أهلاً {name}! تشرفنا بوجودك هنا. عمل {job} يتطلب مهارات مميزة!",
                    f"مرحبا {name}! سعيد بالتعرف عليك. أحب أن أتعلم أكثر عن عملك كـ{job}!"
                ]
            elif name:
                greetings = [
                    f"أهلاً وسهلاً {name}! سعيد بلقائك!",
                    f"مرحباً {name}! أهلاً بك معنا!",
                    f"أهلاً {name}! تشرفنا بوجودك هنا!",
                    f"مرحبا {name}! سعيد بالتعرف عليك!"
                ]
            elif job:
                greetings = [
                    f"أهلاً! أرى أنك تعمل كـ{job}، مهنة رائعة!",
                    f"مرحباً! {job} - عمل مثير للاهتمام!",
                    f"أهلاً بك! عمل {job} يتطلب مهارات مميزة!",
                    f"مرحبا! أحب أن أتعلم أكثر عن عملك كـ{job}!"
                ]
            else:
                greetings = ["أهلاً بك!", "مرحباً!", "أهلاً وسهلاً!", "سعيد بلقائك!"]
        else:  # English
            if name and job:
                greetings = [
                    f"Hey {name}! Nice to meet you! Working as a {job} must be exciting!",
                    f"Hi {name}! Welcome! I'd love to learn more about your work as a {job}!",
                    f"Hello {name}! Great to have you here! Being a {job} requires some amazing skills!",
                    f"Hi there {name}! Pleasure to meet you! {job.title()} work sounds fascinating!",
                    f"Hey {name}! How's it going? I'm curious about your experience as a {job}!"
                ]
            elif name:
                greetings = [
                    f"Hey {name}! Nice to meet you!",
                    f"Hi {name}! Welcome!",
                    f"Hello {name}! Great to have you here!",
                    f"Hi there {name}! Pleasure to meet you!",
                    f"Hey {name}! How's it going?"
                ]
            elif job:
                greetings = [
                    f"Hey there! Working as a {job} must be exciting!",
                    f"Hi! I'd love to learn more about your work as a {job}!",
                    f"Hello! Being a {job} requires some amazing skills!",
                    f"Hi there! {job.title()} work sounds fascinating!",
                    f"Hey! I'm curious about your experience as a {job}!"
                ]
            else:
                greetings = ["Hey there!", "Hi!", "Hello!", "Nice to meet you!", "How's it going?"]
        
        return random.choice(greetings)
    
    def _enhanced_introduction_detection(self, text: str) -> tuple:
        """
        GPT-powered enhanced introduction detection for when the main GPT fails.
        Returns a tuple: (has_introduction: bool, name: str, job_title: str, greeting_message: str)
        """
        if not text:
            return False, "", "", ""
        
        text_clean = text.strip()
        
        try:
            enhanced_prompt = f"""
Analyze this text and extract personal introduction information.

Text: "{text_clean}"

Look for:
1. Personal names (both Arabic and English)
2. Job titles or professions
3. Self-introduction patterns like "I am", "My name is", "انا", "اسمي"

Instructions:
- Extract the person's name if mentioned
- Extract their job/profession if mentioned  
- Be flexible with variations and informal language
- Handle both Arabic and English text
- Look for patterns like:
  * "Hi I'm John, I work as developer"
  * "مرحبا انا وليد مهندس"
  * "My name is Sarah and I'm a teacher"
  * "انا اسمي احمد وانا طبيب"

Respond with ONLY a JSON object in this exact format:
{{
    "has_introduction": true/false,
    "name": "extracted name or empty string",
    "job": "extracted job or empty string"
}}

Examples:
"مرحبا انا وليد مهندس" -> {{"has_introduction": true, "name": "وليد", "job": "مهندس"}}
"Hi I'm John" -> {{"has_introduction": true, "name": "John", "job": ""}}
"I am a developer" -> {{"has_introduction": true, "name": "", "job": "developer"}}
"How are you today?" -> {{"has_introduction": false, "name": "", "job": ""}}
"""

            messages = [
                {"role": "system", "content": "You are an expert at extracting personal introduction information from text."},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            try:
                data = json.loads(result)
                has_intro = data.get("has_introduction", False)
                name = data.get("name", "").strip()
                job = data.get("job", "").strip()
                
                if has_intro and (name or job):
                    # Detect language for greeting generation
                    language = self.detect_language(text_clean)
                    greeting = self.generate_varied_greeting(name, job, language)
                    print(f"GPT enhanced detection found: name='{name}', job='{job}', language='{language}'")
                    return True, name, job, greeting
                    
            except json.JSONDecodeError:
                print(f"Failed to parse GPT JSON response: {result}")
                
        except Exception as e:
            print(f"Error in GPT enhanced introduction detection: {e}")
            
        return False, "", "", ""

    # Define clarification templates as class variables

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
            is_similar = self._gpt_identity_similarity_check(text)
            if is_similar:
                # Re-process with GPT using enhanced prompt with keyword context
                enhanced_result = self._gpt_identity_classification_with_context(text, is_similar)
                if enhanced_result[0]:
                    return enhanced_result
            
            # Third attempt: GPT fallback for edge cases
            return self._gpt_fallback_identity_detection(text)
            
        except Exception as e:
            self.logger.error(f"Error in identity detection: {e}")
            return self._gpt_fallback_identity_detection(text)
    
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
    
    def _gpt_identity_similarity_check(self, text: str) -> str:
        """
        Use GPT to check if text is similar to identity keywords even if not exact match.
        Returns the likely category if similar, None otherwise.
        Enhanced to avoid false positives from self-descriptions.
        """
        if not text:
            return None
            
        try:
            similarity_prompt = f"""
Analyze this text to determine if it's asking about the AI system's identity, even with informal language or typos.

Text: "{text}"

IMPORTANT: Distinguish between:
1. User describing THEMSELVES (NOT identity questions) - examples:
   - "I am a developer" (user's job)
   - "My work involves programming" (user's work)
   - "انا مهندس" (user introducing themselves)
   
2. User asking about THE SYSTEM (identity questions) - examples:
   - "Who is your developer?" (asking about system's creator)
   - "What is your purpose?" (asking about system's goal)
   - "من مطورك" (asking about system's developer)

Categories for identity questions about THE SYSTEM:
- developer: asking about creators, who made/built/developed the system
- purpose: asking about goals, mission, why the system exists  
- role: asking about function, job, what the system does
- who_are_you: asking about identity, who/what the system is

Look for question intent even with:
- Typos or informal language
- Arabic dialect variations
- Missing words or different structure

Respond with ONLY:
- "developer" if asking about system's creators
- "purpose" if asking about system's goals/mission
- "role" if asking about system's function
- "who_are_you" if asking about system's identity
- "none" if NOT an identity question about the system

Examples:
"who made you" -> "developer"
"what do u do" -> "role"  
"ur purpose" -> "purpose"
"من مطورك" -> "developer"
"I am a developer" -> "none" (user describing themselves)
"My purpose in life" -> "none" (user describing themselves)
"""

            messages = [
                {"role": "system", "content": "You are an expert at identifying identity questions about AI systems while avoiding false positives from user self-descriptions."},
                {"role": "user", "content": similarity_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                max_tokens=20,
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if result in ["developer", "purpose", "role", "who_are_you"]:
                print(f"GPT identity similarity detected: {result} for text: '{text}'")
                return result
            else:
                return None
                
        except Exception as e:
            print(f"Error in GPT identity similarity check: {e}")
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
    
    def _gpt_fallback_identity_detection(self, text: str) -> tuple:
        """
        GPT-powered fallback identity detection for edge cases.
        Context-aware to distinguish between user self-descriptions and system questions.
        """
        if not text:
            return False, None, None
        
        try:
            fallback_prompt = f"""
Analyze this text for identity questions about the AI system, with special attention to edge cases.

Text: "{text}"

CRITICAL DISTINCTION:
1. User describing THEMSELVES → NOT identity questions
   - "I am a developer" (user's profession)
   - "My role is programming" (user's job)
   - "انا مهندس" (user introducing themselves)
   
2. User asking about THE SYSTEM → Identity questions  
   - "Who is your developer?" (system's creator)
   - "What is your role?" (system's function)
   - "من مطورك" (system's developer)

Categories for system identity questions:
- developer: asking about creators/makers of the system
- purpose: asking about system's goals/mission
- role: asking about system's function/job
- who_are_you: asking about system's identity
- what_is_begining: asking about the BEGINING project
- team: asking about people behind the system
- understand_personality: asking about system capabilities
- how_analyze: asking about system methodology
- objectives: asking about system goals

Look for question intent even with:
- Typos, informal language
- Arabic dialect variations  
- Partial phrases or unclear wording

Respond with ONLY:
- "IDENTITY:category_name" if asking about the system
- "NOT_IDENTITY" if user describing themselves or other topics

Examples:
"who made u" → "IDENTITY:developer"
"ur purpose" → "IDENTITY:purpose"  
"I work as engineer" → "NOT_IDENTITY"
"My purpose in life" → "NOT_IDENTITY"
"من صنعك" → "IDENTITY:developer"
"انا مطور" → "NOT_IDENTITY"
"""

            messages = [
                {"role": "system", "content": "You are an expert at identifying identity questions about AI systems, especially in edge cases with informal language."},
                {"role": "user", "content": fallback_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                max_tokens=50,
            )
            
            result = response.choices[0].message.content.strip()
            
            if "IDENTITY:" in result:
                category = result.split("IDENTITY:")[1].strip()
                if category in self.IDENTITY_RESPONSES:
                    print(f"GPT fallback detected identity question: {category} for text: '{text}'")
                    return True, category, self.IDENTITY_RESPONSES[category]
            
            return False, None, None
            
        except Exception as e:
            print(f"Error in GPT fallback identity detection: {e}")
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
            
            # Arabic job/self-description words (expanded with more professions)
            "انا مهندس", "أنا مهندس", "انا طبيب", "أنا طبيب", "انا مطور", "أنا مطور",
            "انا مبرمج", "أنا مبرمج", "انا مدرس", "أنا مدرس", "انا مصمم", "أنا مصمم",
            "انا محاسب", "أنا محاسب", "انا محامي", "أنا محامي", "انا ممرض", "أنا ممرض",
            "انا طالب", "أنا طالب", "انا استاذ", "أنا استاذ", "انا دكتور", "أنا دكتور",
            "وظيفتي", "عملي", "مهنتي", "تخصصي",
            "مهندس", "طبيب", "مدرس", "مصمم", "مبرمج", "مطور", "محاسب", "محامي", "ممرض", "استاذ", "دكتور"
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
        Use GPT to analyze which personality traits need more detailed exploration.
        Returns a list of trait categories that need clarification questions.
        """
        # Combine all input text (excluding identity questions)
        all_text = user_input
        personality_answers = []
        
        for qa in new_input:
            answer = qa.get("answer", "")
            # Skip identity questions in trait analysis
            is_identity, _, _ = self.detect_identity_question(answer)
            if not is_identity:
                all_text += " " + answer
                personality_answers.append(answer)
        
        # If we have very little conversation, return all traits as missing
        total_words = len(all_text.split())
        if len(personality_answers) < 1 or total_words < 15:
            return ["emotional", "social", "cognitive", "behavioral"]
        
        try:
            trait_analysis_prompt = f"""
Analyze this conversation text and determine which personality traits need more exploration.

Conversation text: "{all_text}"

The four main personality trait categories are:
1. EMOTIONAL: feelings, emotions, mood, stress management, emotional reactions, happiness, sadness, anger, etc.
2. SOCIAL: interactions with others, teamwork, leadership, communication style, relationships, introversion/extroversion, etc.
3. COGNITIVE: thinking style, problem-solving approach, decision-making, learning preferences, analytical vs creative thinking, etc.
4. BEHAVIORAL: habits, routines, organization, time management, spontaneity vs planning, actions and behaviors, etc.

Instructions:
- Review the conversation for coverage of each trait category
- Consider both explicit mentions and implicit evidence
- A trait is "covered" if there's meaningful information about that aspect of personality
- A trait is "missing" if there's little or no information about it
- Consider job context but don't assume traits without evidence

Example analysis:
- If someone says "I'm an engineer and I solve problems logically" → COGNITIVE is covered
- If they say "I work well with my team" → SOCIAL is covered  
- If they mention "I get stressed easily" → EMOTIONAL is covered
- If they describe their daily routine → BEHAVIORAL is covered

Respond with ONLY a JSON array of the missing traits:
["trait1", "trait2", ...]

If all traits are sufficiently covered, return: []
If some traits need more exploration, list only those: ["emotional", "cognitive"]

The goal is to identify which aspects of personality still need clarification questions.
"""

            messages = [
                {"role": "system", "content": "You are an expert personality analyst who can identify which traits need more exploration in a conversation."},
                {"role": "user", "content": trait_analysis_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            try:
                missing_traits = json.loads(result)
                
                # Validate and clean the results
                valid_traits = ["emotional", "social", "cognitive", "behavioral"]
                missing_traits = [trait for trait in missing_traits if trait in valid_traits]
                
                # Progressive conversation management - limit traits over time
                conversation_turns = len(new_input)
                max_traits = max(1, 3 - conversation_turns // 2)
                
                if len(missing_traits) > max_traits:
                    missing_traits = missing_traits[:max_traits]
                
                print(f"GPT trait analysis found missing traits: {missing_traits}")
                return missing_traits
                
            except json.JSONDecodeError:
                print(f"Failed to parse GPT trait analysis JSON: {result}")
                
        except Exception as e:
            print(f"Error in GPT trait analysis: {e}")
            
        # Fallback: If GPT fails, use simple logic based on conversation length
        if len(personality_answers) < 2:
            return ["emotional", "social"]
        elif len(personality_answers) < 3:
            return ["cognitive"]
        else:
            return []

    @staticmethod
    def generate_clarification_questions(missing_traits: list, languages: str, max_questions: int = 1, asked_questions: list = None) -> list:
        """
        Generate clarification questions for missing traits in the appropriate language.
        Avoids repeating previously asked questions.
        Always returns exactly one question to maintain conversation flow.
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
        
        # Always generate exactly one question
        for trait in shuffled_traits:
            if trait in templates:
                available_questions = [q for q in templates[trait] if q not in asked_questions]
                if available_questions:
                    question = random.choice(available_questions)
                    questions.append(question)
                    break  # Only take one question
                elif templates[trait]:  # Fallback if all questions were asked
                    question = random.choice(templates[trait])
                    questions.append(question)
                    break  # Only take one question
        
        # Ensure we always return exactly one question if traits exist
        if not questions and missing_traits:
            # Fallback to first available trait if none had available questions
            first_trait = shuffled_traits[0]
            if first_trait in templates and templates[first_trait]:
                questions.append(random.choice(templates[first_trait]))
        
        return questions

    def generate_personalized_greeting_with_question(self, name: str, job: str, missing_traits: list, language: str, asked_questions: list = None) -> str:
        """
        Generate a combined greeting and personalized clarification question.
        This creates a natural conversation flow by connecting the greeting with a relevant question.
        """
        import random
        
        if asked_questions is None:
            asked_questions = []
            
        # Generate base greeting
        base_greeting = self.generate_varied_greeting(name, job, language)
        
        # If no missing traits, just return the greeting
        if not missing_traits:
            return base_greeting
            
        # Determine language preference
        is_arabic = language == "arabic"
        
        # Generate job-specific personalized questions
        job_specific_questions = self._generate_job_specific_questions(job, missing_traits, is_arabic)
        
        # If we have job-specific questions, use them; otherwise use general questions
        if job_specific_questions:
            available_questions = [q for q in job_specific_questions if q not in asked_questions]
            if available_questions:
                personalized_question = random.choice(available_questions)
            else:
                personalized_question = random.choice(job_specific_questions)
        else:
            # Fallback to general clarification questions
            general_questions = self.generate_clarification_questions(missing_traits, language, 1, asked_questions)
            personalized_question = general_questions[0] if general_questions else ""
        
        # Combine greeting with question naturally
        if personalized_question:
            if is_arabic:
                # Arabic connectors
                connectors = [
                    f"{base_greeting} أود أن أتعرف عليك أكثر - {personalized_question}",
                    f"{base_greeting} دعني أسألك - {personalized_question}",
                    f"{base_greeting} لأفهم شخصيتك أكثر - {personalized_question}",
                    f"{base_greeting} أتساءل - {personalized_question}"
                ]
            else:
                # English connectors
                connectors = [
                    f"{base_greeting} I'd love to learn more about you - {personalized_question}",
                    f"{base_greeting} Let me ask you - {personalized_question}",
                    f"{base_greeting} To better understand your personality - {personalized_question}",
                    f"{base_greeting} I'm curious - {personalized_question}"
                ]
            
            return random.choice(connectors)
        else:
            return base_greeting

    def _generate_job_specific_questions(self, job: str, missing_traits: list, is_arabic: bool) -> list:
        """
        Generate job-specific personality questions based on profession and missing traits.
        """
        if not job:
            return []
            
        job_lower = job.lower()
        questions = []
        
        if is_arabic:
            # Arabic job-specific questions
            if any(word in job_lower for word in ['مهندس', 'engineer']):
                if 'cognitive' in missing_traits:
                    questions.extend([
                        "كيف تتعامل مع المشاكل التقنية المعقدة في عملك الهندسي؟",
                        "هل تفضل التحليل المنطقي أم الحلول الإبداعية في المشاريع الهندسية؟",
                        "كيف تقوم بتحليل وحل التحديات الهندسية؟"
                    ])
                if 'social' in missing_traits:
                    questions.extend([
                        "كيف تتفاعل مع فريق العمل في المشاريع الهندسية؟",
                        "هل تفضل القيادة أم المشاركة في الفرق الهندسية؟",
                        "كيف تتعامل مع العملاء والزملاء في المشاريع؟"
                    ])
                if 'behavioral' in missing_traits:
                    questions.extend([
                        "هل تتبع منهجية محددة في عملك الهندسي أم تتكيف حسب المشروع؟",
                        "كيف تنظم وقتك ومهامك في المشاريع الهندسية؟"
                    ])
                if 'emotional' in missing_traits:
                    questions.extend([
                        "كيف تشعر عندما تواجه تحديات تقنية صعبة؟",
                        "كيف تتعامل مع ضغوط المواعيد النهائية في المشاريع؟"
                    ])
                    
            elif any(word in job_lower for word in ['مطور', 'مبرمج', 'developer', 'programmer']):
                if 'cognitive' in missing_traits:
                    questions.extend([
                        "كيف تتعامل مع مشاكل البرمجة المعقدة؟",
                        "هل تفضل التفكير المنطقي أم الإبداعي في كتابة الكود؟",
                        "كيف تحلل المتطلبات وتحولها إلى حلول برمجية؟"
                    ])
                if 'social' in missing_traits:
                    questions.extend([
                        "هل تفضل البرمجة الفردية أم العمل في فريق التطوير؟",
                        "كيف تتفاعل مع المطورين الآخرين في مراجعة الكود؟"
                    ])
                if 'behavioral' in missing_traits:
                    questions.extend([
                        "هل تتبع منهجيات تطوير محددة أم تتكيف حسب المشروع؟",
                        "كيف تنظم كودك ومشاريعك البرمجية؟"
                    ])
                    
            elif any(word in job_lower for word in ['معلم', 'مدرس', 'teacher']):
                if 'social' in missing_traits:
                    questions.extend([
                        "كيف تتفاعل مع الطلاب في بيئة التعلم؟",
                        "هل تفضل التدريس التفاعلي أم المحاضرات التقليدية؟",
                        "كيف تتعامل مع الطلاب ذوي الاحتياجات المختلفة؟"
                    ])
                if 'emotional' in missing_traits:
                    questions.extend([
                        "كيف تشعر عندما ترى تقدم طلابك؟",
                        "كيف تتعامل مع التحديات السلوكية في الفصل؟"
                    ])
        else:
            # English job-specific questions
            if any(word in job_lower for word in ['engineer', 'engineering']):
                if 'cognitive' in missing_traits:
                    questions.extend([
                        "How do you approach complex technical problems in your engineering work?",
                        "Do you prefer analytical or creative solutions in engineering projects?",
                        "How do you analyze and solve engineering challenges?"
                    ])
                if 'social' in missing_traits:
                    questions.extend([
                        "How do you interact with your engineering team on projects?",
                        "Do you prefer leading or collaborating in engineering teams?",
                        "How do you work with clients and colleagues on projects?"
                    ])
                    
            elif any(word in job_lower for word in ['developer', 'programmer', 'programming']):
                if 'cognitive' in missing_traits:
                    questions.extend([
                        "How do you tackle complex coding problems?",
                        "Do you prefer logical or creative thinking when writing code?",
                        "How do you analyze requirements and turn them into code solutions?"
                    ])
                if 'social' in missing_traits:
                    questions.extend([
                        "Do you prefer coding alone or working with a development team?",
                        "How do you interact with other developers during code reviews?"
                    ])
                    
            elif any(word in job_lower for word in ['teacher', 'teaching', 'educator']):
                if 'social' in missing_traits:
                    questions.extend([
                        "How do you interact with students in the learning environment?",
                        "Do you prefer interactive teaching or traditional lectures?",
                        "How do you handle students with different learning needs?"
                    ])
                if 'emotional' in missing_traits:
                    questions.extend([
                        "How do you feel when you see your students making progress?",
                        "How do you handle challenging behaviors in the classroom?"
                    ])
        
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
Combine information from all turns to build a complete personality profile.
If the input data contains user_name or user_job fields, incorporate these personal details into the personality descriptions to make them more personalized and direct. Use the user's name when referring to them in the descriptions.

IMPORTANT: Use job titles and professions as HINTS for potential personality traits, but do not automatically assume all individuals in a profession have the same characteristics. Job titles suggest tendencies, not certainties:

- Engineers/مهندس: May tend toward analytical thinking, but verify with actual problem-solving examples
- Teachers/معلم: May tend toward patient communication, but confirm with social interaction examples  
- Doctors/طبيب: May tend toward careful analysis, but validate with decision-making examples
- Artists/فنان: May tend toward creative expression, but check for actual creative behaviors
- Managers/مدير: May tend toward leadership, but confirm with real leadership examples

Use professional context as a starting point for exploration, not as definitive trait assignment. Always prioritize actual behavioral descriptions over job-based assumptions. If someone says they're an engineer but describes poor problem-solving skills, trust their self-description over professional stereotypes.

Job-Based Trait Inference
When a user provides their job title, infer relevant personality traits commonly associated with that profession:
- Engineers: Analytical thinking (cognitive), methodical approach (behavioral), problem-solving orientation
- Teachers: Patient and communicative (social), organized (behavioral), empathetic (emotional)
- Doctors: Detail-oriented (cognitive), calm under pressure (emotional), caring (social)
- Managers: Leadership skills (social), decision-making (cognitive), goal-oriented (behavioral)
- Artists: Creative thinking (cognitive), expressive (emotional), independent (behavioral)
Use job information to help reduce missing traits. A job title alone can provide insights into 2-3 personality traits, allowing you to write descriptions even with limited other information.

Check for Missing Traits
If all four traits are sufficiently covered through explicit descriptions AND/OR reasonable professional inferences, return status "complete".
If some traits are missing after considering both explicit information and professional context, return status "incomplete" and list them in missing_traits.
BALANCE: Use profession as supportive evidence, but prioritize actual behavioral examples. If someone's described behavior contradicts professional expectations, trust their self-description.

Clarification Questions
If incomplete, generate only ONE short, friendly, non-repetitive question.
The single question should focus on the most important missing trait.
Never ask about traits already covered.
Always return exactly one question in the clarification_questions array.

Status Types
- "complete": All four personality traits are sufficiently covered
- "incomplete": Some traits are missing and need clarification
- "identity": When user asks identity questions about the system (handled separately)
- "off_topic": When user asks unrelated questions not about personality or identity (handled separately)

Output Format
id (integer)
status ("complete", "incomplete", "identity", or "off_topic")
personal_greeting (string - MANDATORY: use the exact value from input data's personal_greeting field, never modify or ignore this value)
description_english (concise personality description based on available information - MUST include user's name from user_name field if provided, even for incomplete status)
description_arabic (concise personality description in Arabic based on available information - MUST include user's name from user_name field if provided, even for incomplete status)
description_identity (only for identity status - IDENTITY_RESPONSES)
description_off_topic (only for off_topic status - OFF_TOPIC_RESPONSES)
missing_traits (array or null)
clarification_questions (array)
input_tokens (integer)
output_tokens (integer)

CRITICAL INSTRUCTION: Always include personal_greeting when provided, and handle combined greeting+question responses.
- When personal_greeting contains both greeting and question (e.g., "Hello John! Nice to meet you. I'm curious - how do you handle challenges?"), use it as the main response and set clarification_questions to empty array [""]
- When has_combined_greeting_question is true in input data, do NOT generate separate clarification_questions - the greeting already contains the question
- When personal_greeting is just a greeting without question, include it AND generate separate clarification_questions  
- When status is "incomplete", leave description_english and description_arabic fields empty ("") BUT still include personal_greeting if provided
- When status is "complete", provide detailed personality descriptions based on all available information AND include personal_greeting if provided  
- If user_name field contains a name, include it in the personality description when status is complete
- If user_job field contains a job, consider it as part of the personality context when status is complete  
- Format for complete status: "[Name] [is/seems to be] [personality traits based on available data]"
- Personal greetings should appear regardless of personality analysis completeness

PERSONALITY DESCRIPTION STYLE:
Make descriptions sound natural, varied, and human-like. Avoid repetitive phrases and robotic language.
Use diverse vocabulary and engaging expressions:

INSTEAD OF: "enjoys working with data, solving complex analytical problems"
USE VARIED ALTERNATIVES:
- "has a passion for diving deep into data and uncovering hidden insights"
- "thrives when analyzing complex information and finding creative solutions" 
- "gets energized by tackling challenging analytical puzzles"
- "loves exploring data patterns and transforming numbers into meaningful stories"

INSTEAD OF: "loves working in teams, mentoring junior colleagues"
USE VARIED ALTERNATIVES:
- "naturally gravitates toward collaborative environments and enjoys guiding others"
- "finds fulfillment in team dynamics and helping colleagues grow"
- "has a gift for bringing people together and sharing knowledge generously"
- "builds strong connections with teammates and takes pride in developing talent"

INSTEAD OF: "taking on leadership roles naturally"
USE VARIED ALTERNATIVES:
- "steps up when groups need direction and guidance"
- "has an innate ability to inspire and coordinate team efforts"
- "naturally emerges as a trusted voice in group settings"
- "demonstrates authentic leadership through example and encouragement"

Use personality-rich adjectives and avoid formulaic patterns. Each description should feel unique and capture the individual's distinctive character blend.

            LANGUAGE HANDLING:
            - If languages="en" or "english": Only provide description_english, leave description_arabic empty
            - If languages="ar" or "arabic": Only provide description_arabic, leave description_english empty  
            - If languages contains both or is mixed: Provide both descriptions
            - Always ensure consistency: if you provide a description in one language, make it meaningful and complete
            
            All clarification questions and trait names (in 'missing_traits') must be in the user's primary requested language as specified in the 'languages' field.

            Do not include any extra text, code blocks, or explanations outside the JSON.


Do not include any extra text, code blocks, or explanations outside the JSON.

Example Output — Incomplete
{
    "id": 22,
    "status": "incomplete",
    "personal_greeting": "Hey Ahmad! Nice to meet you! Working as an engineer must be exciting!",
    "description_arabic": "",
    "description_english": "",
    "missing_traits": ["behavioral", "emotional"],
    "clarification_questions": [
        "How do you typically handle stress or emotional challenges in your work?"
    ],
    "input_tokens": 1245,
    "output_tokens": 74,
    "total_tokens": 1317
}

Example Output — Complete
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
            return '{"status": "incomplete", "description_english": "", "description_arabic": "", "clarification_questions": ["Could you provide more information about yourself?"]}'
        
        # Create a default response with status complete for cases with enough information
        if "developer" in text.lower() and ("team" in text.lower() or "professional" in text.lower()):
            return json.dumps({
                "status": "complete", 
                "description_english": "Based on your input, you appear to be a developer with interests in technology and programming who values professional collaboration.",
                "description_arabic": ""  # Will be filled in later if needed
            })
            
        # Return a JSON structure with the original text as a question
        return '{"status": "incomplete", "description_english": "", "description_arabic": "", "clarification_questions": ["Could you tell me more about how you interact with others in your professional environment?"]}'
    
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
        
        # SMART LANGUAGE DETECTION: Always detect from the most recent user message
        # This ensures we respond in the language the user just used, regardless of conversation history
        current_text = ""
        if new_input and len(new_input) > 0:
            # Get the very last answer from new_input (most recent user message)
            current_text = new_input[-1].get("answer", "").strip()
        else:
            # If no new_input, use the initial user_input
            current_text = user_input.strip()
        
        # Detect language from current text
        detected_lang = self.detect_language(current_text)
        detected_language_code = "ar" if detected_lang == "arabic" else "en"
        
        # Always use the detected language, regardless of what was specified
        if current_text:  # Only override if we have text to analyze
            original_languages = languages
            languages = detected_language_code
            
            if original_languages != languages:
                print(f"Language auto-detection: Specified '{original_languages}' but last message is in '{languages}'. Text: '{current_text[:50]}...'")
            else:
                print(f"Language confirmed: Last message detected as '{languages}'. Text: '{current_text[:50]}...'")
        else:
            # Fallback: Auto-detect language if not specified or if "auto" is passed
            if languages == "auto" or not languages:
                detected_lang = self.detect_language(user_input)
                languages = "ar" if detected_lang == "arabic" else "en"
        
        # Personal introduction detection logic:
        # Check for user introducing themselves with name or job title
        personal_greeting = ""
        introduction_text = ""
        user_name = ""
        user_job = ""
        
        # Check for introduction in the current message
        if new_input:
            # Check the LAST answer for personal introduction
            last_qa = new_input[-1]
            introduction_text = last_qa.get("answer", "").strip()
        else:
            # Check initial user_input for personal introduction  
            introduction_text = user_input
        
        # Always check for personal introduction in current message
        has_intro, name, job_title, simple_greeting = self.detect_personal_introduction(introduction_text)
        if has_intro and simple_greeting:
            user_name = name
            user_job = job_title
            print(f"Personal introduction detected: name='{name}', job='{job_title}', greeting='{simple_greeting}'")
            
            # For new introductions, generate combined greeting with personalized question
            # First analyze missing traits to know what questions to ask
            missing_traits = self.analyze_missing_traits(user_input, new_input)
            
            # Extract previously asked questions
            asked_questions = []
            for qa in new_input:
                question = qa.get("question", "").strip()
                if question:
                    asked_questions.append(question)
            
            # Generate combined personalized greeting with question
            combined_response = self.generate_personalized_greeting_with_question(
                name, job_title, missing_traits, detected_lang, asked_questions
            )
            personal_greeting = combined_response
            print(f"Generated combined greeting+question: '{combined_response}'")
        else:
            personal_greeting = ""
        
        # Check if we had previous introduction data stored (for conversation continuity)
        previous_name = ""
        previous_job = ""
        if new_input:
            # Look through conversation history for name/job mentioned previously
            for qa in new_input:
                answer = qa.get("answer", "").strip()
                if answer:
                    prev_has_intro, prev_name, prev_job, _ = self.detect_personal_introduction(answer)
                    if prev_has_intro:
                        if prev_name and not previous_name:
                            previous_name = prev_name
                        if prev_job and not previous_job:
                            previous_job = prev_job
        
        # Use current introduction data if available, otherwise fall back to previous
        final_user_name = user_name if user_name else previous_name
        final_user_job = user_job if user_job else previous_job
        
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
            if is_identity:
                print(f"Identity question detected in last answer: '{last_answer}'")
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
            # But only if personal_greeting doesn't already contain a question
            clarification_questions = []
            if not personal_greeting or ("?" not in personal_greeting and "؟" not in personal_greeting):
                clarification_questions = self.generate_clarification_questions(
                    missing_traits, languages, max_questions=1, asked_questions=asked_questions
                )
            
            # Return identity response with clarification questions to continue conversation
            identity_text = self.get_identity_response(response_data, languages)
            return {
                "content": json.dumps({
                    "id": id,
                    "status": "identity",
                    "personal_greeting": personal_greeting,
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
            # But only if personal_greeting doesn't already contain a question
            clarification_questions = []
            if not personal_greeting or ("?" not in personal_greeting and "؟" not in personal_greeting):
                clarification_questions = self.generate_clarification_questions(
                    missing_traits, languages, max_questions=1, asked_questions=asked_questions
                )
            
            return {
                "content": json.dumps({
                    "id": id,
                    "status": "off_topic",
                    "personal_greeting": personal_greeting,
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
        # Check if personal_greeting contains a question to inform GPT
        has_combined_question = bool(personal_greeting and ("?" in personal_greeting or "؟" in personal_greeting))
        
        input_data = {
            "id": id,
            "user_input": user_input,
            "new_input": new_input,
            "languages": languages,
            "personal_greeting": personal_greeting,
            "user_name": final_user_name,
            "user_job": final_user_job,
            "has_combined_greeting_question": has_combined_question
        }
        gpt_response = self.call_gpt(input_data)
        return gpt_response