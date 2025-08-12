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

    # update memory (optional, keep if needed)
    memory = load_user_memory(req.id)
    memory["user_input"] = (memory.get("user_input", "") + " " + req.user_input.strip()).strip()
    memory["new_input"] = (memory.get("new_input", "") + " " + req.get_combined_new_input().strip()).strip()
    save_user_memory(req.id, memory["user_input"], memory["new_input"])

    # Prepare input for analyzer
    try:
        gpt_json = analyzer.analyze(
            id=req.id,
            user_input=req.user_input,
            new_input=req.new_input,
            languages=req.languages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer error: {e}")

    # The model's output is in gpt_json['content'] as a JSON string; parse it
    import json
    try:
        model_output = json.loads(gpt_json["content"])
    except Exception:
        raise HTTPException(status_code=500, detail={"error": "Invalid JSON from GPT", "raw_response": gpt_json})

    # Attach token usage if present
    if "input_tokens" in gpt_json:
        model_output["input_tokens"] = gpt_json["input_tokens"]
    if "output_tokens" in gpt_json:
        model_output["output_tokens"] = gpt_json["output_tokens"]
    if "total_tokens" in gpt_json:
        model_output["total_tokens"] = gpt_json["total_tokens"]

    # Ensure 'id' is always an integer for response validation
    try:
        model_output["id"] = int(model_output["id"])
    except Exception:
        model_output["id"] = req.id

    # Enforce language output: only fill in requested language(s)
    requested_langs = req.languages
    if isinstance(requested_langs, str):
        requested_langs = [requested_langs]
    requested_langs = [l.lower() for l in requested_langs]
    if not ("en" in requested_langs or "english" in requested_langs):
        model_output["description_english"] = ""
    if not ("ar" in requested_langs or "arabic" in requested_langs):
        model_output["description_arabic"] = ""

    return model_output
