"""
Simple test API for just the PersonalityAnalyzer to avoid dependency issues
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.personality_analyzer import PersonalityAnalyzer

app = FastAPI()

# Initialize only the analyzer (skip predictor to avoid dependency issues)
analyzer = PersonalityAnalyzer()

class UserRequest(BaseModel):
    id: int
    user_input: str
    new_input: List[dict] = []  # Each dict: {"question": str, "answer": str}
    languages: Union[str, List[str], None] = None

@app.post("/analyze-personality")
async def analyze_personality(req: UserRequest):
    """Test endpoint for personality analysis."""
    
    print(f"Processing request for user ID: {req.id}")
    print(f"User input: {req.user_input[:100]}...")
    print(f"Conversation history: {len(req.new_input)} exchanges")
    
    # Call analyzer
    result = analyzer.analyze(
        id=req.id,
        user_input=req.user_input,
        new_input=req.new_input,
        languages=req.languages or "en"
    )
    
    print(f"Result has identity response: {bool(result.get('description_identity'))}")
    
    return result

@app.get("/")
async def root():
    return {"message": "Test API for PersonalityAnalyzer context-aware identity detection"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
