# 🧠 DALL IN — Personality AI Chatbot

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/beginingCompany/dall-in-model)
[![License](https://img.shields.io/badge/license-proprietary-red)](#license)
[![Model](https://img.shields.io/badge/model-BERT%20%7C%20XLMRobertaModel)](#model-details)
[![Language](https://img.shields.io/badge/language-Arabic%20%7C%20English-yellowgreen)](#)

---
**Data Source:** Dr. Ibrahim Mohamed Ahmed Hussain  
---

## 🔍 Overview

**DALL IN** is a symbolic AI chatbot designed to interpret personality input and return structured profiles. It uses the **BEGINING Scale**, a symbolic framework that represents psychological dimensions through a 3-letter code.

The model outputs:

- A symbolic code (`letter`)
- Academic and career recommendations
- Hobby suggestions
- Descriptions and strengths in **Arabic & English**
- Cognitive and emotional profiling

---

## 📐 BEGINING Scale

### 🔹 Concept

The **BEGINING Scale**, invented by **Dr. Ibrahim Mohamed Ahmed Hussain** in 2004, is a symbolic system for understanding:

- Leadership style  
- Emotional & social intelligence  
- Key cognitive strengths  
- Human productivity profiles

It generates one of **120 symbolic 3-letter codes**, tested on over **6,000 individuals** and refined over 15+ years.

> **Authorship Declaration**  
> I, Ibrahim Mohamed Ahmed Hussain, affirm that I am the sole creator of the BEGINING Scale.  
> The model has been developed based on research and real-world testing to simulate structured symbolic intelligence.

---

## 🎯 Objectives

- Symbolic, AI-compatible personality analysis  
- Support for **academic, professional**, and **behavioral** guidance  
- A foundation for AI systems with symbolic cognition  
- Real-time inference from user input (text-based)

---


🧠 Project Structure

Dall-IN-MODEL/
├── config/
│   └── paths.py                     # Centralized path configuration
├── app/
│   ├── __init__.py
│   ├── api.py                       # FastAPI main route handler
│   ├── GPT_api.py                   # Optional OpenAI integration
│   └── predict.py                   # API logic for symbolic prediction
├── data/
│   ├── raw/                         # Provided raw data
│   │   ├── majors.csv
│   │   └── result_symbols.csv
│   └── processed/                   
│       ├── cleaned_data.csv
│       └── BIGINING_dataset.csv
├── models/
│   ├── classifier/
│   │   ├── classifier.pt            # custom classifier head (PyTorch)
│   │   ├── config.json              # XLM-RoBERTa backbone config
│   │   └── model.safetensors        # XLM-RoBERTa backbone weights (Hugging Face format)
│   └── tokenizer/
│       ├── sentencepiece.bpe.model
│       ├── special_tokens_map.json
│       └── tokenizer_config.json
├── src/
│   ├── utils/
│   │   ├── data_loader.py
│   │   └── augmentations.py
│   ├── training/
│   │   └── trainer.py
│   ├── train.py                     # Training entrypoint
│   └── predict.py                   # CLI prediction interface
├── .gitignore
├── README.md
└── requirements.txt


⚙️ How to Use
1. Install Dependencies
pip install -r requirements.txt

2. Run FastAPI Server
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000

3. Send Input for Prediction
curl -X POST "http://127.0.0.1:8000/predict" \
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
