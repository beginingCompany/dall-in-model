import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
from typing import Optional, Union, List
from app.predict import PersonalityPredictor
from app.personality_analyzer import PersonalityAnalyzer
from app.apply_question_repetition_fix import apply_fix
from app.emergency_question_fix import apply_emergency_fix

app = FastAPI()

# Apply the question repetition fixes
apply_fix()
apply_emergency_fix()

predictor = PersonalityPredictor(num_labels=120, top_k=3)
analyzer = PersonalityAnalyzer()

# Simple in-memory user memory storage
_user_memory_store = {}

class PredictionRequest(BaseModel):
    text: str

class PredictionItem(BaseModel):
    class_name: str
    confidence: str

class PredictionResponse(BaseModel):
    text: str
    predictions: List[PredictionItem]

class UserRequest(BaseModel):
    id: int
    user_input: str
    new_input: List[dict] = []  # Each dict: {"question": str, "answer": str}
    languages: Union[str, List[str], None] = None

    @classmethod
    def coerce_languages(cls, v):
        if v is None:
            return ["en"]
        if isinstance(v, str):
            return [v.lower().strip()]
        if isinstance(v, list):
            return [str(item).lower().strip() for item in v]
        raise ValueError("languages must be a string or list of strings")

    def get_languages(self):
        return self.coerce_languages(self.languages)

    def get_combined_new_input(self) -> str:
        """
        Combine all 'answer' fields from new_input list into a single string.
        """
        if not self.new_input:
            return ""
        # Accept both old and new formats for backward compatibility
        if isinstance(self.new_input, list) and all(isinstance(item, dict) and ("answer" in item) for item in self.new_input):
            return "\n".join(str(item.get("answer", "")).strip() for item in self.new_input if item.get("answer"))
        return str(self.new_input)

class TraitResponse(BaseModel):
    id: int
    status: str
    description_arabic: Optional[str] = ""
    description_english: Optional[str] = ""
    missing_traits: Optional[List[str]] = ""
    clarification_questions: Optional[List[str]] = [""]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

# In-memory functions to simulate DB
def load_user_memory(user_id: int):
    return _user_memory_store.get(user_id, {"user_input": "", "new_input": ""})

def save_user_memory(user_id: int, user_input: str, new_input: str):
    _user_memory_store[user_id] = {"user_input": user_input, "new_input": new_input}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        df_result = predictor.predict([request.text])
        result = df_result.iloc[0]
        preds = [
            PredictionItem(
                class_name=p.get("class_name") or p.get("class") or p.get("label", ""),
                confidence=p.get("confidence", "0%")
            )
            for p in result['predictions']
        ]
        return PredictionResponse(text=result['text'], predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/analyze-personality", response_model=TraitResponse)
async def analyze_personality(request: Request):
    t0 = time.time()
    try:
        data = await request.json()
        req = UserRequest(**data)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input JSON")

    # update memory 
    memory = load_user_memory(req.id)
    memory["user_input"] = (memory.get("user_input", "") + " " + req.user_input.strip()).strip()
    memory["new_input"] = (memory.get("new_input", "") + " " + req.get_combined_new_input().strip()).strip()
    save_user_memory(req.id, memory["user_input"], memory["new_input"])

    # Ensure we have language(s) as a list
    languages = req.get_languages()
    if isinstance(languages, str):
        languages = [languages]
        
    try:
        # Build full context from user_input and new_input (Q&A pairs)
        full_context = PersonalityAnalyzer.build_full_context(req.user_input, req.new_input)
        gpt_json = analyzer.analyze(full_context, "", languages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer error: {e}")

    result = {
        "id": req.id,
        "input_tokens": gpt_json.get("input_tokens"),
        "output_tokens": gpt_json.get("output_tokens"),
        "total_tokens": gpt_json.get("total_tokens"),
    }

    if "description_arabic" in gpt_json or "description_english" in gpt_json:
        # Check if descriptions are actually provided and contain real descriptions (not questions)
        eng_desc = gpt_json.get("description_english", "").strip()
        ar_desc = gpt_json.get("description_arabic", "").strip()
        
        has_english = eng_desc and len(eng_desc) > 0
        has_arabic = ar_desc and len(ar_desc) > 0
        
        # Check if the "description" is actually a question (common issue)
        is_eng_question = has_english and (eng_desc.endswith("?") or "could you" in eng_desc.lower() or "can you" in eng_desc.lower())
        is_ar_question = has_arabic and ar_desc.endswith("؟")
        
        # Languages requested by the user
        langs = req.get_languages()
        needs_english = any(lang in ["english", "en"] for lang in langs)
        needs_arabic = any(lang in ["arabic", "ar"] for lang in langs)
        
        # Only mark as complete if the requested language descriptions are provided and are not questions
        is_complete = True
        if (needs_english and (not has_english or is_eng_question)) or (needs_arabic and (not has_arabic or is_ar_question)):
            is_complete = False
            
        # If descriptions are actually questions, add them to clarification_questions
        clarification_questions = []
        if not is_complete:
            if is_eng_question and needs_english:
                clarification_questions.append(eng_desc)
            if is_ar_question and needs_arabic:
                clarification_questions.append(ar_desc)
                
        result.update({
            "status": "complete" if is_complete else "incomplete",
            "description_arabic": ar_desc if has_arabic and not is_ar_question else "",
            "description_english": eng_desc if has_english and not is_eng_question else "",
            "clarification_questions": clarification_questions if clarification_questions else None
        })
    elif "missing_traits" in gpt_json or "clarification_questions" in gpt_json:
        # If status is incomplete but no clarification questions provided, add a default question
        clarification_questions = gpt_json.get("clarification_questions")
        if gpt_json.get("status") == "incomplete" and not clarification_questions:
            clarification_questions = ["Could you tell me more about your personality traits? For example, how do you think and approach problems, or how do you interact with others?"]
            
        result.update({
            "status": "incomplete",
            "description_arabic": gpt_json.get("description_arabic", ""),
            "description_english": gpt_json.get("description_english", ""),
            "missing_traits": gpt_json.get("missing_traits"),
            "clarification_questions": clarification_questions
        })
    elif "clarification_prompt" in gpt_json or "clarification_prompts" in gpt_json:
        questions = []
        if "clarification_prompts" in gpt_json and gpt_json["clarification_prompts"]:
            questions = gpt_json["clarification_prompts"]
        elif "clarification_prompt" in gpt_json and gpt_json["clarification_prompt"]:
            questions = [gpt_json["clarification_prompt"]]
        result.update({
            "status": "incomplete",
            "description_arabic": gpt_json.get("description_arabic", ""),
            "description_english": gpt_json.get("description_english", ""),
            "missing_traits": None,
            "clarification_questions": questions
        })
    else:
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected GPT output", "raw_response": gpt_json},
        )

    # Final check: If status is incomplete but no questions provided, add a generic question
    if result.get("status") == "incomplete" and (not result.get("clarification_questions") or result.get("clarification_questions") is None):
        if full_context and isinstance(full_context, str) and "software developer" in full_context.lower():
            result["clarification_questions"] = ["Could you tell me more about how you approach problems and organize your work as a developer?"]
        else:
            result["clarification_questions"] = ["Could you share more about your personality traits? For example, how do you typically react to challenges or interact with others?"]
    
    return result
