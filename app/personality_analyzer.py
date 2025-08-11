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
    def generate_clarification_prompt(user_input: str) -> str:

        TRAIT_PATTERNS = {
            "emotional": r"enthusiastic|happy|sad|calm|feel(s)?|emotion|stress|excited|passion|motivat(ed|ion)|anxious|worry|nerv(ous|ousness)|joy|depress(ed|ion)|angry|anger|fear|afraid|content|satisf(ied|action)|frustrated|overwhelm(ed|ing)|lonely|love|hate|mood|temper|relax(ed|ing)|tense|anxiet(y|ies)|cry|tears|laugh|giggle|smile|frown|bitter|sweet|tender|sensitive|numb|reactive|respond|respon(se|sive)|upset|disturb(ed|ance)|irritat(ed|ion)|hope(ful|less)|optimis(t|tic|m)|pessimis(t|tic|m)|gloomy|bright|cheerful|melanchol(y|ic)|nostalg(ia|ic)|sentimental|proud|shame|guilt|embarrass(ed|ment)|jealous|envy|inspir(e|ed|ation)|awe|wonder|amaz(e|ed)|thrill(ed|ing)|excite(d|ment)|bore(d|dom)|interest(ed|ing)|curious|apathy|indifferen(t|ce)|compassion|empath(y|etic|ize)",
            
            "social": r"collaborative|team|help|assist|shy|introvert|extrovert|interact|polite|friendly|people|others|social|group|communit(y|ies)|network|connect(ed|ion)|relation(s|ship)|friend(s|ly|ship)|colleague|famil(y|iar)|partner|spouse|parent|child(ren)|sibling|acquaintance|neighbor|leader(ship)|follow(er)|mentor|teach(er|ing)|student|learn(er|ing)|service|volunteer|assist(ant|ance)|support(ive|er)|care(ing|giver|taker)|trust(ing|worthy|ed)|loyal(ty)|honest(y)|reliable|depend(able|ent)|independent|autonomous|solitary|alone|isolat(ed|ion)|withdraw(n|al)|reserved|quiet|loud|vocal|outspoken|assertive|passive|aggressive|persuasive|convincing|diplomatic|tactful|blunt|direct|gentle|harsh|kind|cruel|fair|unfair|just|unjust|bias(ed)|prejudice|discriminat(e|ion|ory)|tolera(nt|nce)|patient|impatient|understand(ing)|misunderstand(ing)|communicat(e|ion|or|ive)|listen(er|ing)|talk(ative|er)|express(ive|ion)|articulate|eloquent|stammering|fluent|sharing|generou(s|sity)|selfish|give(r)|take(r)|reciprocat(e|ion)|exchange|barter|trade|negotiate|compromise|conflict|argument|debate|discuss(ion)|dialogue|monologue|silence|noise(y)|crowd(ed)|busy|popular|unpopular|like(able|d)|dislike(d)|admir(e|ed|ation)|respect(ed|ful)|disrespect(ed|ful)",
            
            "cognitive": r"think|critical|logical|analytical|understand|reason|solve|strateg(y|ic)|intuitive|intellect(ual|ually)|smart|intelligent|genius|bright|dull|slow|quick|fast|sharp|clever|wise|foolish|rational|irrational|objective|subjective|bias(ed)|prejudice|open-minded|close-minded|dogmatic|flexible|rigid|adaptable|learn|memorize|forget|recall|remember|study|read|write|compute|calculate|measure|assess|evaluate|judge|decide|choose|select|prioritize|rank|order|classify|categorize|organize|system(atic|atize)|method(ical)|process|procedure|algorithm|heuristic|creative|imaginative|innovative|original|conventional|traditional|visionary|vision|foresight|hindsight|insight|introspect(ive|ion)|reflect(ive|ion)|contemplate|meditate|ponder|consider|examine|investigate|explore|discover|invent|design|develop|build|construct|engineer|architect|plan|project|forecast|predict|anticipate|expect|assume|presume|conclude|infer|deduce|induce|abstract|concrete|literal|figurative|symbolic|metaphor(ic|ical)|analogy|compare|contrast|similar|different|pattern|trend|association|correlation|causation|effect|impact|influence|perceive|perception|sense|sensation|observe|observation|attention|focus|concentrate|distract(ed|ion)|daydream|imagine|visualize|conceptualize|theorize|hypothesize|experiment|test|verify|validate|prove|disprove|skeptic(al|ism)|doubt|question|answer|problem|solution|issue|concern|matter|topic|subject|field|discipline|expert(ise)|specialize|generalize|broad|narrow|deep|shallow|complex|simple|complicated|straightforward|clear|unclear|confusing|ambiguous|precise|accurate|exact|approximate|estimate|guess|speculate|wonder|curious|interest(ed)|bore(d)|apathetic|practical|theoretical|abstract|concrete|realistic|idealistic|pragmatic|utopian|cynical|optimistic|pessimistic",
            
            "behavioral": r"organized|spontaneous|routine|habit|act|impulsive|disciplined|methodical|child(ish)?|behav(e|ior)|conduct|perform|do|practice|engage|participate|active|passive|react|respond|automatic|conscious|unconscious|habit(ual)|routine|ritual|pattern|custom|tradition|convention|norm|standard|rule|law|principle|value|belief|attitude|opinion|view|stance|position|perspective|outlook|philosophy|ideology|religion|spiritual(ity)|moral(s|ity)|ethic(s|al)|honest|dishonest|integrity|corrupt(ion)|virtue|vice|sin|righteous|wicked|good|bad|evil|angel(ic)|devil(ish)|demon(ic)|diligent|lazy|industrious|idle|productive|unproductive|efficient|inefficient|effective|ineffective|useful|useless|helpful|unhelpful|beneficial|harmful|constructive|destructive|create|destroy|build|demolish|conserve|waste|save|spend|earn|invest|profit|loss|succeed|fail|win|lose|achieve|accomplish|complete|finish|start|begin|initiate|continue|persist|persevere|endure|quit|abandon|surrender|give up|try|attempt|effort|struggle|strive|aspire|ambition|goal|target|aim|objective|purpose|mission|vision|dream|wish|desire|want|need|crave|yearn|long|hope|expect|anticipate|plan|prepare|ready|unready|organize|disorganize|mess(y)|neat|tidy|clean|dirty|precise|imprecise|accurate|inaccurate|careful|careless|cautious|reckless|safe|dangerous|risk(y)|adventure|explore|discover|travel|journey|path|road|direction|way|method|process|procedure|system|structure|framework|schedule|calendar|agenda|appointment|meeting|gathering|party|celebration|ritual|ceremony|habit|pattern|cycle|rhythm|tempo|pace|speed|slow|fast|quick|gradual|sudden|abrupt|smooth|rough|easy|difficult|hard|simple|complex|complicated|challenge|obstacle|barrier|hurdle|problem|solution|fix|repair|maintain|improve|enhance|upgrade|advance|progress|regress|decline|worsen|deteriorate|decay|age|grow|develop|mature|immature|childish|adult|responsible|irresponsible|dependable|undependable|reliable|unreliable|trustworthy|untrustworthy|honest|dishonest|loyal|disloyal|faithful|unfaithful|committed|uncommitted|dedicated|undedicated|devoted|undevoted|passionate|passionless|enthusiastic|unenthusiastic|excited|unexcited|interested|uninterested|engaged|disengaged|involved|uninvolved|attentive|inattentive|focused|unfocused|concentrated|unconcentrated|distracted|alert|aware|conscious|unconscious|asleep|awake|active|inactive|passive|energetic|lethargic|vigorous|weak|strong|powerful|powerless|assertive|unassertive|aggressive|unaggressive|competitive|uncompetitive|ambitious|unambitious|driven|undriven|motivated|unmotivated|inspired|uninspired|determined|undetermined|resolved|unresolved|decided|undecided|certain|uncertain|sure|unsure|confident|unconfident|secure|insecure|comfortable|uncomfortable|content|discontent|satisfied|unsatisfied|happy|unhappy|joyful|joyless|glad|sad|pleased|displeased|delighted|disappointed|excited|unexcited|thrilled|unthrilled|ecstatic|depressed"
        }

        CLARIFICATION_TEMPLATES = {
            "emotional": [
                "How do you usually feel in difficult or exciting situations? What emotions come up and how do you handle them?",
                "Would you describe yourself as more emotionally expressive or reserved? How do your emotions typically influence your decisions?",
                "When facing stress or pressure, what emotional responses do you notice in yourself?",
                "How would your close friends describe your emotional temperament? Are you even-keeled, passionate, or somewhere in between?",
                "Do you find it easy to recognize and process your feelings, or do you tend to set emotions aside?",
                "What kinds of situations or events tend to bring out strong emotional responses in you?",
                "How quickly do your moods change? Do you tend to stay in one emotional state for long periods?",
                "When you're feeling down, what strategies do you use to regulate your emotions?",
                "How comfortable are you expressing vulnerability or difficult emotions to others?"
            ],
            "social": [
                "Can you describe how you typically interact with others—do you enjoy helping, leading, or prefer to work alone?",
                "In social settings, do you tend to initiate conversations or prefer others to approach you first?",
                "How would you compare your behavior in small groups versus larger gatherings?",
                "What role do you usually take when working in teams or group projects?",
                "How important is maintaining a wide social network to you compared to having a few close relationships?",
                "How do you typically respond to conflict or disagreement with others?",
                "Do you find it energizing or draining to spend time with others? How much alone time do you need?",
                "How would you describe your communication style? Are you more direct or diplomatic?",
                "What qualities do you value most in your friendships and relationships?",
                "How easily do you trust others and open up about personal matters?"
            ],
            "cognitive": [
                "What kind of thinking comes naturally to you? Are you analytical, imaginative, or more intuitive in decisions?",
                "When solving problems, do you prefer to follow established methods or create new approaches?",
                "How do you typically gather and process information before making decisions?",
                "Do you tend to focus more on details or the big picture when analyzing situations?",
                "How comfortable are you with theoretical or abstract concepts versus concrete, practical matters?",
                "What kinds of mental challenges do you find most engaging or stimulating?",
                "How would you describe your learning style? Do you prefer visual information, hands-on experience, or something else?",
                "When faced with a complex decision, what process do you typically follow?",
                "How important is creativity in your thinking process? In what ways do you express creativity?",
                "How do you typically approach unfamiliar or ambiguous situations that require thinking on your feet?"
            ],
            "behavioral": [
                "Tell me about your habits or actions—do you prefer routines, act on impulse, or stay flexible?",
                "How much structure and planning do you typically incorporate into your daily life?",
                "When starting new projects, do you dive right in or carefully plan each step?",
                "How would you describe your approach to time management and deadlines?",
                "Are you more likely to stick with established methods or experiment with new ways of doing things?",
                "How easily do you adapt when plans change unexpectedly?",
                "What personal habits or routines are most important to your productivity or well-being?",
                "Do you tend to be more cautious and careful or bold and risk-taking in your actions?",
                "How persistent are you when facing obstacles or setbacks?",
                "How would you describe your energy levels throughout a typical day? When are you most active?"
            ]
        }

        import random
        
        text = user_input.lower()
        present = []
        for trait, pattern in TRAIT_PATTERNS.items():
            if re.search(pattern, text):
                present.append(trait)

        missing = [trait for trait in TRAIT_PATTERNS if trait not in present]

        if not missing:
            return ""
        elif len(missing) == 1:
            # Randomly select one question from the list of questions for this trait
            return random.choice(CLARIFICATION_TEMPLATES[missing[0]])
        else:
            # For multiple missing traits, select one question from each category
            selected_questions = [random.choice(CLARIFICATION_TEMPLATES[t]) for t in missing]
            # Return 1-2 questions maximum to avoid overwhelming the user
            return " ".join(selected_questions[:2])

    SYSTEM_PROMPT = """
You are an AI assistant that generates rich, professional, and expressive personality descriptions in English and Arabic for each user, based on all historical and current inputs linked to their unique `id`.

---

**Memory & Context Retention:**  
For each `id`, store and recall all past user inputs. This includes both `user_input` and `new_input` answers from previous sessions. Use this complete history to create accurate, personalized, and narrative personality descriptions that remain consistent over time.

---

**Strict Field Usage Rule:**
- The fields "description_english" and "description_arabic" must only contain a personality description. Never include a question or request for more information in these fields.
- If you need to ask for more information, use the "clarification_questions" field only.
- IMPORTANT: Even with minimal information, ALWAYS provide at least a brief personality description based on whatever limited traits you can identify from the user's input. Never return empty description fields when status is "complete".

**User Input (JSON):**  
- `id`: unique user identifier (integer or string)  
- `user_input`: main free-text input from the user  
- `new_input`: optional array of `{question, answer}` pairs from clarification rounds  
- `languages`: optional, e.g., `"en"`, `"ar"`, `"both"` (default = `"both"`)  

---

**Personality Construction Rules:**  
1. **Information Sources:** Combine all historical and current inputs for the `id`.  
2. **Key Personality Aspects to Cover:**  
   - Emotional & stress responses (e.g., calm under pressure, anxious in new settings)  
   - Social interaction style (e.g., enjoys teamwork, prefers solitude)  
   - Thinking/problem-solving approach (e.g., methodical planner, fast decision-maker)  
   - Day-to-day behavior & habits (e.g., organized, spontaneous)  
3. **Skills Integration:** Seamlessly embed all mentioned technical, professional, or practical skills (e.g., programming, writing, leadership) in both the English and Arabic descriptions.  
4. **Style:** Narrative format (not bullet points) that reads naturally but is still concise and clear. Match tone and flow similar to these **examples from BIGINING_dataset.csv**:  

---

**Example (English):**  
*"John is a calm and methodical thinker who thrives under pressure. He enjoys collaborating in teams but also values focused solo work when problem-solving. Naturally organized, he prefers to plan his day in advance, ensuring efficiency in both personal and professional settings. Skilled in Python programming and data analysis, John often applies structured approaches to tackle complex challenges."*

**Example (Arabic):**  
*"جون شخص هادئ ويفكر بطريقة منهجية ويزدهر تحت الضغط. يستمتع بالعمل ضمن فريق لكنه يقدر أيضًا العمل الفردي المركّز عند حل المشكلات. بطبيعته منظم، يفضل تخطيط يومه مسبقًا لضمان الكفاءة في حياته الشخصية والمهنية. ماهر في برمجة بايثون وتحليل البيانات، وغالبًا ما يستخدم أساليب منظمة للتعامل مع التحديات المعقدة."*

---

**Language Rules:**  
- If `languages` = `"en"`, return only `description_english`.  
- If `languages` = `"ar"`, return only `description_arabic`.  
- If `languages` is missing or `"both"`, return both descriptions.  

---

**Clarification Process:**  
- On first interaction for an `id`, thank the user for what they shared and reference a detail from their input.  
- On later clarifications, vary encouragement phrases and avoid repeating identical thank-you wording.  
- Always read the full conversation history and **never** ask about a trait already addressed.  
- If an answer is vague, encourage more detail without forcing it.  
- If the user expresses frustration, acknowledge, show empathy, and either summarize or move to a new question.  
- The goal is to gather all four aspects with enough richness to build a complete description — once they are covered, stop asking for more unless the user explicitly requests an update.  

---

**Clarification Question Generation:** 
- Generate 1–2 short, friendly, non-repetitive questions in the user's selected language(s) to gather missing information. 
- Always personalize by referencing something the user has already shared.  
- Provide 1–2 short, friendly, non-repetitive questions in the user’s selected language(s).  
- Example:  
  ```json
  {"clarification_questions": ["That's interesting that you enjoy programming. Could you share how you usually handle unexpected challenges?", "How do you typically spend your weekends?"]}

**all examples in this prompt just to show case not a pattren**
Output Format (JSON only, no extra text):
When all four aspects are covered:
{
  "id": <same as input>,
  "status": "complete",
  "description_english": "Full, detailed personality description covering emotional, social, cognitive, and behavioral traits",
  "description_arabic": "Arabic translation of the full personality description",
  "missing_traits": [],
  "clarification_questions": [],
  "input_tokens": <int>,
  "output_tokens": <int>,
  "total_tokens": <int>
}
When aspects are missing:
{
  "id": <same as input>,
  "status": "incomplete",
  "missing_traits": ["social_interaction", "daily_habits"],
  "clarification_questions": ["..."],
  "description_english": "Partial personality description based on the available information",
  "description_arabic": "Arabic translation of the partial personality description",
  "input_tokens": <int>,
  "output_tokens": <int>,
  "total_tokens": <int>
}

CRITICAL: Never return empty description fields if you mark status as "complete". Always generate at least a brief description based on whatever information you have, even if minimal.
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
        # Try to extract JSON from code block or fallback to braces
        pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1)
        # Fallback: extract from first { to last }
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1:
            return text[json_start:json_end+1]
        return text.strip()

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

    def call_gpt(self, input_text: str, languages: str, id: int, max_tokens: int = 1200) -> dict:
        prompt = json.dumps({
            "user_input": input_text,
            "new_input": "",
            "id": id,
            "languages": languages
        }, ensure_ascii=False)

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
        self.logger.debug(f"GPT response: {content[:200]}...")

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }

    def analyze(
        self,
        user_input: str,
        new_input: str = "",
        languages: List[str] = ["english"],
        id: int = 1
    ) -> dict:
        self.logger.debug(f"Starting analysis for user {id}")
        # Normalize languages to 'en', 'ar', or 'both'
        normalized_langs = []
        for lang in languages:
            if lang.lower() in ("en", "english"):
                normalized_langs.append("en")
            elif lang.lower() in ("ar", "arabic"):
                normalized_langs.append("ar")
        if not normalized_langs:
            normalized_langs = ["both"]
        lang_param = "both" if len(normalized_langs) > 1 else normalized_langs[0]

        combined_input = self.combine_inputs_safely(user_input, new_input)
        self.logger.debug(f"Combined input length: {len(combined_input)} characters")
        gpt_response = self.call_gpt(combined_input, lang_param, id=id)
        json_text = self.extract_json(gpt_response["content"])

        try:
            gpt_json = json.loads(json_text)
        except json.JSONDecodeError:
            self.logger.error(f"GPT returned invalid JSON: {gpt_response}")
            raise RuntimeError(f"GPT returned invalid JSON: {gpt_response['content']}")

        # Attach token usage
        gpt_json["input_tokens"] = gpt_response.get("input_tokens")
        gpt_json["output_tokens"] = gpt_response.get("output_tokens")
        gpt_json["total_tokens"] = gpt_response.get("total_tokens")

        # Ensure we always have descriptions if status is complete
        if gpt_json.get("status") == "complete":
            # If descriptions are empty but status is complete, generate fallback descriptions
            if "en" in normalized_langs and not gpt_json.get("description_english"):
                gpt_json["description_english"] = "Based on limited information, this person appears to have some distinctive personality traits that would benefit from further exploration."
            
            if "ar" in normalized_langs and not gpt_json.get("description_arabic"):
                gpt_json["description_arabic"] = "بناءً على معلومات محدودة، يبدو أن هذا الشخص لديه بعض سمات الشخصية المميزة التي قد تستفيد من مزيد من الاستكشاف."
        
        # If descriptions are empty, mark as incomplete and ask clarifying questions
        elif (
            ("en" in normalized_langs and not gpt_json.get("description_english")) or
            ("ar" in normalized_langs and not gpt_json.get("description_arabic"))
        ):
            gpt_json["status"] = "incomplete"
            if not gpt_json.get("clarification_questions"):
                gpt_json["clarification_questions"] = [
                    "Could you tell me more about yourself? This will help me create a more accurate personality description.",
                    "What aspects of your personality do you consider most distinctive or important?"
                ]

        return gpt_json

