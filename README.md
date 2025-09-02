# 🧠 DALL IN — Personality AI Chatbot

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/beginingCompany/dall-in-model)
[![License](https://img.shields.io/badge/license-BEGINING-red)]
[![Model](https://img.shields.io/badge/model-BERT%20%7C%20XLMRobertaModel)]
[![Language](https://img.shields.io/badge/language-Arabic%20%7C%20English-yellowgreen)]

______________________________________________________________________

## **Data Source:** Dr. Ibrahim Mohamed Ahmed Hussain

## 🔍 Overview

```json
            ┌───────────────┐
            │   User Input   │
            └───────┬───────┘
                    │
        ┌───────────┴───────────┐
        │ Greeting / Off-topic? │
        └───────┬───────┬───────┘
                │Yes    │No
                ▼       │
    ┌───────────────────┐
    │ Fill personal_    │
    │ greeting_and_off  │
    │ Status = incomplete
    │ Missing = all     │
    │ Ask 1 clarification│
    └───────────────────┘
                        ▼
            ┌───────────┴────────────┐
            │ Identity Question?     │
            └───────┬───────┬────────┘
                    │Yes    │No
                    ▼       │
    ┌───────────────────────┐
    │ Fill description_     │
    │ identity              │
    │ Status = incomplete   │
    │ Missing = all         │
    │ Ask 1 clarification   │
    └───────────────────────┘
                            ▼
              ┌─────────────┴──────────────┐
              │ Personality Answer?        │
              └───────────┬────────────────┘
                          │Yes
                          ▼
            ┌──────────────────────────────┐
            │ Extract traits: emotional,    │
            │ social, cognitive, behavioral │
            └───────────┬──────────────────┘
                        ▼
             ┌──────────┴───────────┐
             │ All traits complete? │
             └───────┬───────┬──────┘
                     │Yes    │No
                     ▼       ▼
    ┌───────────────────┐   ┌───────────────────┐
    │ Status = complete  │   │ Status = incomplete│
    │ Descriptions filled│   │ List missing traits│
    │ No missing traits  │   │ Ask 1 clarification│
    │ No clarification   │   └───────────────────┘
    └───────────────────┘
```

**DALL IN** is a symbolic AI chatbot designed to interpret personality input and return structured profiles. It uses the **BEGINING Scale**, a symbolic framework that represents psychological dimensions through a 3-letter code.

The model outputs:

- A symbolic code (`letter`)
- Academic and career recommendations
- Hobby suggestions

## 📝 Input Processing

The system includes an `input_processor` module to handle and format input data for personality analysis. This processor:

- Combines the initial user input with question-answer pairs
- Processes multiple questions in a single entry
- Formats the data for optimal analysis

### Usage Example

```python
from app.input_processor import format_for_analysis

# Sample input data
data = {
    "id": 123,
    "user_input": "I am a software developer",
    "new_input": [
        {
            "question": "How do you handle stress at work?",
            "answer": "I take short breaks to clear my mind."
        }
    ],
    "languages": "en"
}

# Process the data
processed_data = format_for_analysis(data)

# The processed_data can now be sent to the analyzer
```

### API Integration

The input processor is automatically used when making requests to the `/analyze-personality` endpoint, ensuring all input is properly formatted before analysis.

- Descriptions and strengths in **Arabic & English**
- Cognitive and emotional profiling

______________________________________________________________________

## 📐 BEGINING Scale

### 🔹 Concept

The **BEGINING Scale**, invented by **Dr. Ibrahim Mohamed Ahmed Hussain** in 2004, is a symbolic system for understanding:

- Leadership style
- Emotional & social intelligence
- Key cognitive strengths
- Human productivity profiles

It generates one of **120 symbolic 3-letter codes**, tested on over **6,000 individuals** and refined over 15+ years.

> **Authorship Declaration**\
> I, Ibrahim Mohamed Ahmed Hussain, affirm that I am the sole creator of the BEGINING Scale.\
> The model has been developed based on research and real-world testing to simulate structured symbolic intelligence.

______________________________________________________________________

## 🎯 Objectives

- Symbolic, AI-compatible personality analysis
- Support for **academic, professional**, and **behavioral** guidance
- A foundation for AI systems with symbolic cognition
- Real-time inference from user input (text-based)

______________________________________________________________________

🧠 Project Structure

Dall-IN-MODEL/
├── config/
│ └── paths.py # Centralized path configuration
├── app/
│ ├── **init**.py
│ ├── api.py # FastAPI main route handler
│ ├── GPT_api.py # Optional OpenAI integration
│ └── predict.py # API logic for symbolic prediction
├── data/
│ ├── raw/ # Provided raw data
│ │ ├── majors.csv
│ │ └── result_symbols.csv
│ └── processed/\
│ ├── cleaned_data.csv
│ └── BIGINING_dataset.csv
├── models/
│ ├── classifier/
│ │ ├── classifier.pt # custom classifier head (PyTorch)
│ │ ├── config.json # XLM-RoBERTa backbone config
│ │ └── model.safetensors # XLM-RoBERTa backbone weights (Hugging Face format)
│ └── tokenizer/
│ ├── sentencepiece.bpe.model
│ ├── special_tokens_map.json
│ └── tokenizer_config.json
├── src/
│ ├── utils/
│ │ ├── data_loader.py
│ │ └── augmentations.py
│ ├── training/
│ │ └── trainer.py
│ ├── train.py # Training entrypoint
│ └── predict.py # CLI prediction interface
├── .gitignore
├── README.md
└── requirements.txt

⚙️ How to Use

1. Install Dependencies
   pip install -r requirements.txt

1. Run FastAPI Server
   uvicorn app.api:app --reload --host 127.0.0.1 --port 8000

1. Send Input for Prediction
   curl -X POST "<http://127.0.0.1:8000/predict>" \
   -H "Content-Type: application/json" \
   -d '{"text": "I enjoy working in teams, love psychology, and prefer creative thinking."}'

🔍 Model Details
Architecture: Torch-based classifier using BERT/XLM-R

Tokenizer: SentencePiece multilingual tokenizer

Input: Free-text natural language (Arabic or English)

Output: Symbolic letter + structured profile + JSON

🧑‍💻 Contributors
🧠 Dr. Ibrahim Mohamed Ahmed Hussain
Inventor of the BEGINING Scale (2004)

📜 License
The BEGINING Scale is the intellectual property of Dr. Ibrahim Mohamed Ahmed Hussain

Use of the system or data requires written permission from the respective authors.

**Signature:** ENG Ahmed Almalki AI Engineer
