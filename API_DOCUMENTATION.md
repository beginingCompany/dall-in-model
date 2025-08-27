# BEGINING Personality Analysis API

## Overview
The BEGINING Personality Analysis API provides comprehensive personality trait analysis and classification using advanced machine learning models. The API supports multilingual analysis (English and Arabic) and includes intelligent handling for identity questions and off-topic queries.

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Root Endpoint
**GET /** 
Returns basic API information and available endpoints.

**Response:**
```json
{
  "message": "BEGINING Personality Analysis API",
  "version": "1.0.0",
  "endpoints": {
    "predict": "/predict",
    "analyze_personality": "/analyze-personality", 
    "health": "/health",
    "docs": "/docs"
  }
}
```

### 2. Health Check
**GET /health**
Checks if the API and its services are running properly.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1756299591.0893903,
  "services": {
    "analyzer": "loaded",
    "predictor": "loaded"
  }
}
```

### 3. Personality Analysis (Main Endpoint)
**POST /analyze-personality**
Analyzes user input to extract personality traits and provide intelligent responses.

**Request Body:**
```json
{
  "id": 1,
  "user_input": "I am a happy person who enjoys working with teams",
  "new_input": [
    {
      "question": "How do you handle stress?", 
      "answer": "I try to stay calm and think logically"
    }
  ],
  "languages": "en"
}
```

**Parameters:**
- `id` (integer): Unique user identifier
- `user_input` (string): Initial user input text
- `new_input` (array): Optional conversation history with Q&A pairs
- `languages` (string/array): Language preference ("en", "ar", or ["en", "ar"])

**Response Status Types:**

#### 1. Complete Analysis
```json
{
  "id": 1,
  "status": "complete",
  "description_english": "A person with strong analytical abilities and collaborative social approach.",
  "description_arabic": "",
  "description_identity": "",
  "description_off_topic": "", 
  "missing_traits": [],
  "clarification_questions": [],
  "input_tokens": 869,
  "output_tokens": 93,
  "total_tokens": 960
}
```

#### 2. Incomplete Analysis
```json
{
  "id": 1,
  "status": "incomplete",
  "description_english": "",
  "description_arabic": "",
  "description_identity": "",
  "description_off_topic": "",
  "missing_traits": ["cognitive", "behavioral"],
  "clarification_questions": [
    "How do you approach complex problems?",
    "What are your typical daily habits?"
  ],
  "input_tokens": 250,
  "output_tokens": 45,
  "total_tokens": 295
}
```

#### 3. Identity Questions
```json
{
  "id": 1,
  "status": "identity",
  "description_english": "",
  "description_arabic": "",
  "description_identity": "I'm Minus Zero, part of the BEGINING project — a personality trait measurement system...",
  "description_off_topic": "",
  "missing_traits": ["emotional", "social"],
  "clarification_questions": [
    "How do you typically interact with others?"
  ],
  "input_tokens": 0,
  "output_tokens": 0,
  "total_tokens": 0
}
```

#### 4. Off-Topic Questions
```json
{
  "id": 1,
  "status": "off_topic",
  "description_english": "",
  "description_arabic": "",
  "description_identity": "",
  "description_off_topic": "That's an interesting question, but I'm Minus Zero - a personality analysis system focused on understanding human traits...",
  "missing_traits": ["emotional", "social", "cognitive", "behavioral"],
  "clarification_questions": [
    "What would you like to explore about yourself today?"
  ],
  "input_tokens": 0,
  "output_tokens": 0,
  "total_tokens": 0
}
```

### 4. Personality Classification
**POST /predict**
Provides personality type classification from the 120-type BEGINING system.

**Request Body:**
```json
{
  "text": "I am a creative and analytical person who enjoys teamwork"
}
```

**Response:**
```json
{
  "text": "I am a creative and analytical person who enjoys teamwork",
  "predictions": [
    {
      "class_name": "YRS",
      "confidence": "76.08%"
    },
    {
      "class_name": "YON", 
      "confidence": "21.73%"
    },
    {
      "class_name": "YVN",
      "confidence": "2.19%"
    }
  ]
}
```

### 5. User Memory Management
**GET /user/{user_id}/memory**
Retrieves stored conversation memory for a user.

**Response:**
```json
{
  "user_id": 123,
  "memory": {
    "user_input": "I am happy and social...",
    "new_input": "Q: How do you handle stress? A: I stay calm..."
  },
  "timestamp": 1756299591.0893903
}
```

**DELETE /user/{user_id}/memory**
Clears stored conversation memory for a user.

**Response:**
```json
{
  "message": "Memory cleared for user 123"
}
```

### 6. API Statistics
**GET /stats**
Returns basic usage statistics.

**Response:**
```json
{
  "active_users": 5,
  "total_users_served": 5,
  "timestamp": 1756299591.0893903
}
```

## Features

### 🧠 **Intelligent Analysis**
- **4-Trait System**: Analyzes emotional, social, cognitive, and behavioral traits
- **120 Personality Types**: Classification into BEGINING's comprehensive personality system
- **Adaptive Questioning**: Generates targeted clarification questions for missing traits

### 🌍 **Multilingual Support**
- **English & Arabic**: Full support for both languages
- **Auto-Detection**: Automatic language detection from user input
- **Localized Responses**: Culture-appropriate responses in user's preferred language

### 🎯 **Smart Question Handling**
- **Identity Detection**: Recognizes questions about the system itself
- **Off-Topic Detection**: Identifies unrelated questions and guides back to personality topics
- **Conversation Flow**: Maintains context across multiple interactions

### 🔧 **Robust Error Handling**
- **Graceful Fallbacks**: Provides meaningful responses even when processing fails
- **Input Validation**: Comprehensive validation of request parameters
- **Memory Management**: Persistent conversation history with cleanup options

## Interactive Documentation
Visit `/docs` for Swagger UI documentation or `/redoc` for ReDoc documentation when the server is running.

## Example Usage

### Python Example
```python
import requests

# Analyze personality
response = requests.post(
    "http://localhost:8000/analyze-personality",
    json={
        "id": 1,
        "user_input": "I enjoy helping others and solving creative problems",
        "languages": "en"
    }
)

result = response.json()
print(f"Status: {result['status']}")
print(f"Questions: {result['clarification_questions']}")
```

### PowerShell Example
```powershell
$body = @{
    id = 1
    user_input = "I am social and analytical"
    languages = "en"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/analyze-personality" `
    -Method Post -Body $body -ContentType "application/json"
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/analyze-personality" \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "user_input": "I love working with people and solving problems",
    "languages": "en"
  }'
```

## Status Codes
- **200**: Success
- **400**: Bad Request (invalid input)
- **422**: Validation Error (missing/invalid parameters)
- **500**: Internal Server Error

The API is now complete and fully functional with comprehensive error handling, multilingual support, and intelligent question detection!
