import torch
import pandas as pd
from transformers import XLMRobertaTokenizer
from config.paths import PathConfig
from src.training.trainer import PersonalityClassifierTrainer
import logging
import os
from typing import List, Union, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # Disable TensorFlow warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonalityPredictor:
    """
    Personality code predictor. Supports top-K, probability output, and efficient batch inference.
    """

    def __init__(
        self,
        num_labels: int,
        top_k: int = 3,
        paths: Optional[PathConfig] = None,
    ):
        self.num_labels = num_labels
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths = paths if paths else PathConfig()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.label_map = self._load_label_map()
        logger.info(f"Predictor initialized [labels={self.num_labels}, top_k={self.top_k}, device={self.device}]")

    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = False,
        batch_size: int = 16,
        output_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predicts the top-k classes for input texts.

        Args:
            texts: List of texts or a single string.
            return_probs: Return raw probability for each class in top-k.
            batch_size: Batch size for inference.
            output_file: Optional; save results to this path as CSV.

        Returns:
            pd.DataFrame with columns: text, predictions, remaining_confidence
        """
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list), "texts must be a string or list of strings"

        results = []
        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start+batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                encodings = {k: v.to(self.device) for k, v in encodings.items()}

                outputs = self.model(
                    input_ids=encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                    labels=None
                )
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)

                topk = torch.topk(probs, self.top_k, dim=1)
                topk_indices = topk.indices.cpu().tolist()
                topk_values = topk.values.cpu().tolist()

                for i, text in enumerate(batch_texts):
                    values = topk_values[i]
                    idxs = topk_indices[i]
                    sum_vals = sum(values)
                    predictions = []
                    for idx, raw_val in zip(idxs, values):
                        normalized_conf = raw_val / sum_vals if sum_vals > 0 else 0
                        predictions.append({
                            "class_name": self.label_map.get(idx, f"UNK_{idx}"),
                            "confidence": f"{round(normalized_conf * 100, 2)}%",   # normalized confidence
                            "raw_prob": float(round(normalized_conf * 100, 4)) if return_probs else None
                        })
                    # remaining confidence is zero or very close after normalization
                    results.append({
                        "text": text,
                        "predictions": predictions,
                        "remaining_confidence": None
                    })

        df = pd.DataFrame(results)
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to: {output_file}")
        return df

    def _load_tokenizer(self):
        """Loads tokenizer from disk using Hugging Face format."""
        if not self.paths.TOKENIZER.exists():
            raise FileNotFoundError(f"Tokenizer not found at {self.paths.TOKENIZER}")
        logger.info(f"Loading tokenizer from {self.paths.TOKENIZER}")
        return XLMRobertaTokenizer.from_pretrained(self.paths.TOKENIZER.as_posix())

    def _load_model(self):
        """Loads transformer backbone and classifier head."""
        if not self.paths.MODELS.exists():
            raise FileNotFoundError(f"Model not found at {self.paths.MODELS}")
        logger.info(f"Loading model from {self.paths.MODELS}")
        model = PersonalityClassifierTrainer(self.num_labels).initialize_model()
        model_file = self.paths.MODELS / "classifier" / "classifier.pt"
        if model_file.exists():
            model.load_model(self.paths.MODELS)
        else:
            # Fallback: load state_dict directly if present
            try:
                state_dict = torch.load(self.paths.MODELS, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise RuntimeError(f"Could not load model state_dict: {e}")
        model.eval()
        return model.to(self.device)

    def _load_label_map(self):
        """Builds int->label mapping from processed data."""
        processed_data_path = self.paths.DATA_PROCESSED / "BIGINING_dataset.csv"
        if not processed_data_path.exists():
            raise FileNotFoundError(f"Label map file not found: {processed_data_path}")
        df = pd.read_csv(processed_data_path)
        if "letter" not in df.columns:
            label_col = df.columns[df.dtypes == "object"][0]
            logger.warning(f"'letter' column not found; using '{label_col}' for label mapping.")
            cats = df[label_col].astype('category').cat.categories
        else:
            cats = df['letter'].astype('category').cat.categories
        return {i: label for i, label in enumerate(cats)}

if __name__ == "__main__":
    predictor = PersonalityPredictor(num_labels=120, top_k=3)
    test_texts = [
        "I enjoy taking risks and trying new experiences",
        "I prefer careful planning and structured routines"
    ]
    predictions = predictor.predict(test_texts, return_probs=True)
    print("\nPredictions:")
    print(predictions)
