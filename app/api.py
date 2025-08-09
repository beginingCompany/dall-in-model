import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
from typing import Optional, Union, List
from app.predict import PersonalityPredictor
from app.personality_analyzer import PersonalityAnalyzer

app = FastAPI()

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
    new_input: Optional[str] = ""
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

class TraitResponse(BaseModel):
    id: int
    status: str
    description_arabic: Optional[str] = ""
    description_english: Optional[str] = ""
    missing_traits: Optional[List[str]] = None
    clarification_questions: Optional[List[str]] = None
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

    memory = load_user_memory(req.id)
    memory["user_input"] = (memory.get("user_input", "") + " " + req.user_input.strip()).strip()
    memory["new_input"] = (memory.get("new_input", "") + " " + req.new_input.strip()).strip()
    save_user_memory(req.id, memory["user_input"], memory["new_input"])

    try:
        gpt_json = analyzer.analyze(
            user_input=memory["user_input"],
            new_input=memory["new_input"],
            languages=req.get_languages(),
            id=req.id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer error: {e}")

    result = {
        "id": req.id,
        "input_tokens": gpt_json.get("input_tokens"),
        "output_tokens": gpt_json.get("output_tokens"),
        "total_tokens": gpt_json.get("total_tokens"),
    }

    if "description_arabic" in gpt_json or "description_english" in gpt_json:
        result.update({
            "status": "complete",
            "description_arabic": gpt_json.get("description_arabic", ""),
            "description_english": gpt_json.get("description_english", "")
        })
    elif "missing_traits" in gpt_json or "clarification_questions" in gpt_json:
        result.update({
            "status": "incomplete",
            "description_arabic": gpt_json.get("description_arabic", ""),
            "description_english": gpt_json.get("description_english", ""),
            "missing_traits": gpt_json.get("missing_traits"),
            "clarification_questions": gpt_json.get("clarification_questions")
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

    return result
