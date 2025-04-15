from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent.resolve()

class PathConfig:
    def __init__(self):
        # Define paths
        self.DATA_RAW = BASE_DIR / "dall-in-model/data/raw"
        self.DATA_PROCESSED = BASE_DIR / "dall-in-model/data/processed"
        self.DATA_SYNTHETIC = BASE_DIR / "dall-in-model/data/synthetic"
        self.MODELS = BASE_DIR / "dall-in-model/models"
        self.TOKENIZER = self.MODELS / "tokenizer"
        self.CLASSIFIER = self.MODELS / "classifier"
        self.MODEL_PATH = Path('models/trained_model.pth')
        
        self._create_directories()
        
    def _create_directories(self):
        """Ensure all required directories exist"""
        self.DATA_RAW.mkdir(parents=True, exist_ok=True)
        self.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        self.DATA_SYNTHETIC.mkdir(parents=True, exist_ok=True)
        self.MODELS.mkdir(parents=True, exist_ok=True)
        self.TOKENIZER.mkdir(parents=True, exist_ok=True)
        self.CLASSIFIER.mkdir(parents=True, exist_ok=True)
