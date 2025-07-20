from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
class PathConfig:
    def __init__(self):
        # Define paths
        self.DATA_RAW = BASE_DIR / "data/raw"
        self.DATA_PROCESSED = BASE_DIR / "data/processed"
        self.DATA_SYNTHETIC = BASE_DIR / "data/synthetic"
        self.MODELS = BASE_DIR / "models"
        self.TOKENIZER = self.MODELS / "tokenizer"
        self.CLASSIFIER = self.MODELS / "classifier"
        
        self._create_directories()
        
    def _create_directories(self):
        """Ensure all required directories exist"""
        self.DATA_RAW.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        self.DATA_SYNTHETIC.mkdir(parents=True, exist_ok=True)
        self.MODELS.mkdir(parents=True, exist_ok=True)
        self.TOKENIZER.mkdir(parents=True, exist_ok=True)
        self.CLASSIFIER.mkdir(parents=True, exist_ok=True)
