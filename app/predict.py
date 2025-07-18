import torch
import pandas as pd
from transformers import XLMRobertaTokenizer
from config.paths import PathConfig
from src.training.trainer import PersonalityClassifierTrainer
import logging
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disables tokenizer warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"         # Disables TensorFlow warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonalityPredictor:
    def __init__(self, num_labels: int, top_k=3):  # top_k parameter
        self.num_labels = num_labels
        self.top_k = top_k  # Store as instance variable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths = PathConfig()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.label_map = self._load_label_map()

    def predict(self, texts: list[str], return_probs=False):
        """
        Predicts top-k classes for each input text.
        
        Parameters:
        - texts: List of strings to predict
        - return_probs: If True, return raw probabilities (non-normalized)

        Returns:
        - DataFrame of predictions
        """
        results = []

        with torch.no_grad():
            for text in texts:
                inputs = self.preprocess(text)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)

                topk = torch.topk(probs, self.top_k)
                topk_indices = topk.indices[0].tolist()
                topk_values = topk.values[0].tolist()

                # Normalize selected top-k values to 100%
                total_topk_sum = sum(topk_values)
                normalized_values = [v / total_topk_sum for v in topk_values]

                predictions = []
                for idx, norm_val, raw_val in zip(topk_indices, normalized_values, topk_values):
                    predictions.append({
                        "class": self.label_map[idx],
                        "confidence": f"{round(norm_val * 100, 2)}%",
                        "raw_prob": float(round(raw_val * 100, 2)) if return_probs else None
                    })

                results.append({
                    "text": text,
                    "predictions": predictions,
                    "remaining_confidence": f"{round((1 - total_topk_sum) * 100, 2)}%" if not return_probs else None
                })

        return pd.DataFrame(results)
    
    def _load_tokenizer(self):
        """Load the saved tokenizer"""
        if not self.paths.TOKENIZER.exists():
            raise FileNotFoundError(f"Tokenizer not found at {self.paths.TOKENIZER}")
            
        return XLMRobertaTokenizer.from_pretrained(self.paths.TOKENIZER.as_posix())

    def _load_model(self):
        """Load the trained model"""
        if not self.paths.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {self.paths.MODEL_PATH}")

        # Recreate the model architecture
        model = PersonalityClassifierTrainer(self.num_labels).initialize_model()
#         model.load_state_dict(torch.load(self.paths.MODEL_PATH, map_location=self.device))
        state_dict = torch.load(self.paths.MODEL_PATH, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model.to(self.device)

    def _load_label_map(self):
        """Create label mapping from training data"""
        # Load original training data to get label mapping
        processed_data_path = self.paths.DATA_PROCESSED / "BIGINING_dataset.csv"
        df = pd.read_csv(processed_data_path)
        return {i: label for i, label in enumerate(df['letter'].astype('category').cat.categories)}

    def preprocess(self, text: str):
        """Tokenize input text"""
        return self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

if __name__ == "__main__":
    # usage
    predictor = PersonalityPredictor(num_labels=120)  # number of classes
    
    test_texts = [
        "I enjoy taking risks and trying new experiences",
        "I prefer careful planning and structured routines"
    ]
    
    predictions = predictor.predict(test_texts)
    print("\nPredictions:")
    print(predictions)