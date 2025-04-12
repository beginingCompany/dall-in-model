personality_chatbot/
├── config/
│   └── paths.py
├── data/
│   ├── raw/
│   │   ├── majors.csv
│   │   └── result_symbols.csv
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   └── merged_data.csv
│   └── synthetic/
│       └── synthetic_data.csv
├── models/
│   ├── trained_model.pth
│   └── tokenizer/
│       ├── sentencepiece.bpe.model
│       ├── special_tokens_map.json
│       └── tokenizer_config.json
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── augmentations.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt


    # Identifier
    'letter',
    
    # Majors
    'major_1_arabic', 'major_1_english',
    'major_2_arabic', 'major_2_english',
    
    # Jobs
    'job_1_arabic', 'job_1_english',
    'job_2_arabic', 'job_2_english',
    
    # Hobby
    'hobby_arabic', 'hobby_english',
    
    # Diploma
    'diploma_arabic', 'diploma_english',
    
    # Department
    'department_letter',
    'department_arabic', 'department_english',
    
    # Descriptions
    'description_arabic', 'description_english'
