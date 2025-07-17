import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.data_loader import DataLoader
from src.training.trainer import PersonalityClassifierTrainer
from config.paths import PathConfig
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
paths = PathConfig()

def validate_dataset(df: pd.DataFrame, min_samples_per_class: int = 5):
    class_counts = Counter(df['letter'])
    logger.info(f"Class distribution:\n{class_counts.most_common(5)}...")
    
    problematic = [cls for cls, count in class_counts.items() if count < min_samples_per_class]
    if problematic:
        raise ValueError(
            f"{len(problematic)} classes below {min_samples_per_class} samples: "
            f"{problematic[:5]}..."
        )

def main():
    data_loader = DataLoader(num_synthetic_samples=200)
    
    try:
        logger.info("Starting training pipeline")
        
        # Load and combine data
        original_df = data_loader.load_and_validate()
        synthetic_df = data_loader.generate_synthetic_data()
        
        combined_df = pd.concat([original_df, synthetic_df])
        combined_df = (
            combined_df
            .drop_duplicates(subset=['text'])
            .groupby('letter')
            .filter(lambda x: len(x) >= 2)  # Remove classes with <2 samples
            .reset_index(drop=True)
        )
        
        # Convert to categorical
        combined_df["letter"] = combined_df["letter"].astype("category")
        combined_df["label_code"] = combined_df["letter"].cat.codes
        
        # Final validation
        validate_dataset(combined_df)
        num_classes = len(combined_df['letter'].cat.categories)
        
        # Validate dataset
        logger.info(f"Total samples: {len(combined_df):,}")
        logger.info(f"Number of classes: {num_classes}")

        # Adaptive splitting
        min_split_size = num_classes * 5  # 5 samples per class minimum
        if len(combined_df) >= min_split_size:
            # Stratified split
            train_df, temp_df = train_test_split(
                combined_df,
                test_size=0.3,
                stratify=combined_df['letter'],
                random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                stratify=temp_df['letter'],
                random_state=42
            )
        else:
            # Grouped K-Fold for small datasets
            from sklearn.model_selection import StratifiedKFold
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            indices = list(kf.split(combined_df, combined_df['letter']))
            train_idx, temp_idx = indices[0]
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.2,
                random_state=42
            )
            
            train_df = combined_df.iloc[train_idx]
            val_df = combined_df.iloc[val_idx]
            test_df = combined_df.iloc[test_idx]

        logger.info("\n=== Dataset Splits ==="
            f"\nTrain: {len(train_df):,} samples"
            f"\nValidation: {len(val_df):,} samples"
            f"\nTest: {len(test_df):,} samples")

        # Initialize and train
        trainer = PersonalityClassifierTrainer(num_labels=num_classes)
        model = trainer.train(train_df, val_df, data_loader.tokenizer)

        # Final evaluation
        logger.info("\n--- Final Test Evaluation ---")
        test_metrics = trainer.validate(
            model, 
            trainer._create_loader(test_df, data_loader.tokenizer, is_train=False)
        )
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}% | Loss: {test_metrics['loss']:.4f}")

    except Exception as e:
        logger.exception("Critical training error")
        raise

if __name__ == "__main__":
    main()