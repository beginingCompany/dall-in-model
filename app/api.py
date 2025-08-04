import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError
from typing import Optional, Union, List
from app.predict import PersonalityPredictor
from app.personality_analyzer import PersonalityAnalyzer
from app.db.user_memory_db import create_table, load_user_memory, save_user_memory

app = FastAPI()
predictor = PersonalityPredictor(num_labels=120, top_k=3)

class PredictionRequest(BaseModel):
    text: str

class PredictionItem(BaseModel):
    class_name: str
    confidence: str

class PredictionResponse(BaseModel):
    text: str
    predictions: List[PredictionItem]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        result = predictor.predict([request.text]).iloc[0]
        # Print for debugging
        print(result['predictions'])
        return {
            "text": result['text'],
            "predictions": [
                {
                    "class_name": p.get("class") or p.get("class_name") or p.get("label", ""),
                    "confidence": p.get("confidence", 0.0)
                }
                for p in result['predictions']
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

analyzer = PersonalityAnalyzer()

@app.on_event("startup")
def startup():
    create_table()

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

@app.post("/analyze-personality", response_model=TraitResponse)
async def analyze_personality(request: Request):
    t0 = time.time()
    print("[START] analyze_personality", t0)
    try:
        data = await request.json()
        req = UserRequest(**data)
    except ValidationError as ve:
        print("[ERROR] Validation failed after", time.time() - t0, "sec")
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        print("[ERROR] JSON parse failed after", time.time() - t0, "sec")
        raise HTTPException(status_code=400, detail="Invalid input JSON")

    print("[INFO] After input parse:", time.time() - t0, "sec")
    memory = load_user_memory(req.id)
    print("[INFO] After load_user_memory:", time.time() - t0, "sec")
    memory["user_input"] += " " + req.user_input.strip()
    memory["new_input"] += " " + req.new_input.strip()
    save_user_memory(req.id, memory["user_input"], memory["new_input"])
    print("[INFO] After save_user_memory:", time.time() - t0, "sec")

    try:
        gpt_json = analyzer.analyze(
            user_input=memory["user_input"],
            new_input=memory["new_input"],
            languages=req.get_languages(),
            id=req.id
        )
    except Exception as e:
        print("[ERROR] analyzer.analyze failed after", time.time() - t0, "sec")
        raise HTTPException(status_code=500, detail=str(e))
    print("[INFO] After analyzer.analyze:", time.time() - t0, "sec")

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
        print("[ERROR] Unexpected GPT output after", time.time() - t0, "sec")
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected GPT output", "raw_response": gpt_json},
        )
    print("[END] analyze_personality completed in", time.time() - t0, "sec")
    return result
