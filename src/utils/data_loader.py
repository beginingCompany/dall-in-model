import pandas as pd
import logging
import random
from config.paths import PathConfig
from transformers import XLMRobertaTokenizer
from typing import List, Dict

logger = logging.getLogger(__name__)

paths = PathConfig()

class DataLoader:
    def __init__(self, num_synthetic_samples: int = 20): 
        self.num_synthetic_samples = num_synthetic_samples
        self.required_columns = [
            'letter', 'major_1_arabic', 'major_1_english',
            'job_1_arabic', 'job_1_english', 'hobby_arabic',
            'hobby_english', 'description_arabic', 'description_english',
            'Leadership_Motivation_en', 'Emotional_Social_Intelligence_en',	
            'Key_Strengths_Applications_en', 'Leadership_Motivation_ar',	
            'Emotional_Social_Intelligence_ar',	'Key_Strengths_Applications_ar'

        ]
        self.tokenizer = self._load_tokenizer()
        self.variation_templates = self._init_variation_templates()

    def _init_variation_templates(self) -> Dict[str, List[str]]:
        return {
            'english': [
                "As a {adjective} {job} with background in {major}, I {action}...",
                "My personality ({description}) makes me {reaction}...",
                "{hobby} helps me {benefit}..."
            ],
            'arabic': [
                "كـ {job} {adjective} خلفيتي في {major}، أنا {action}...",
                "شخصيتي ({description}) تجعلني {reaction}...",
                "{hobby} يساعدني في {benefit}..."
            ]
        }

    def _validate_columns(self, df: pd.DataFrame):
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _validate_integrity(self, df: pd.DataFrame):
        """Validate data integrity without class size checks"""
        if df.isnull().values.any():
            raise ValueError("Dataset contains missing values")
        if df['letter'].duplicated().any():
            raise ValueError("Duplicate letters detected")

    def _load_tokenizer(self):
        try:
            # Create directory if it doesn't exist
            paths.TOKENIZER.mkdir(parents=True, exist_ok=True)
            
            # Check for essential tokenizer files
            required_files = ['vocab.json', 'merges.txt', 'tokenizer_config.json']
            has_all_files = all((paths.TOKENIZER / file).exists() for file in required_files)
            
            if has_all_files:
                logger.info("Loading existing tokenizer")
                return XLMRobertaTokenizer.from_pretrained(paths.TOKENIZER.as_posix())
            
            # If files are missing, download fresh copy
            logger.info("Downloading and saving new tokenizer")
            tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
            tokenizer.save_pretrained(paths.TOKENIZER.as_posix())
            return tokenizer

        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {str(e)}")
            raise

    def load_and_validate(self) -> pd.DataFrame:
        DATA_PROCESSED = paths.DATA_PROCESSED / "BIGINING_dataset.csv"
        if not DATA_PROCESSED.exists():
            raise FileNotFoundError(f"BIGINING_dataset not found at {DATA_PROCESSED}")
            
        df = pd.read_csv(DATA_PROCESSED)
        self._validate_columns(df)
        self._validate_integrity(df)
        logger.info(f"Loaded {len(df)} validated samples")
        return df

    def _generate_varied_text(self, row: pd.Series, language: str) -> str:
        template = random.choice(self.variation_templates[language])
        variations = {
            'adjective': random.choice(["experienced", "skilled", "professional"]),
            'action': random.choice(["analyze deeply", "focus on details", "think strategically"]),
            'reaction': random.choice(["thrive in chaos", "prefer structure", "seek challenges"]),
            'benefit': random.choice(["relax", "focus", "be creative"])
        }
        return template.format(
            major=row[f'major_1_{language}'],
            job=row[f'job_1_{language}'],
            hobby=row[f'hobby_{language}'],
            description=row[f'description_{language}'],
            **variations
        )

    def generate_synthetic_data(self) -> pd.DataFrame:
        original_df = self.load_and_validate()
        synthetic = []
        
        for _, row in original_df.iterrows():
            for _ in range(self.num_synthetic_samples):
                # English variations
                synthetic.append({
                    "text": self._generate_varied_text(row, 'english'),
                    "letter": row["letter"]
                })
                
                # Arabic variations
                synthetic.append({
                    "text": self._generate_varied_text(row, 'arabic'),
                    "letter": row["letter"]
                })

        synth_df = pd.DataFrame(synthetic)
        
        # Ensure uniqueness and validity
        synth_df = (
            synth_df.drop_duplicates(subset=['text'])
            .dropna()
            .pipe(lambda df: df[df['text'].str.strip() != ""])
        )
        
        logger.info(f"Generated {len(synth_df)} unique synthetic samples")
        return synth_df