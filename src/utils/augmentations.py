import nlpaug.augmenter.word as naw
import logging
import random
from typing import List
from torch.utils.data import Dataset, DataLoader
from config.paths import PathConfig
import torch
logger = logging.getLogger(__name__)
paths = PathConfig()

class TextAugmenter:
    def __init__(self, aug_p: float = 0.3):
        self.aug_p = aug_p
        self.augmenters = {
            'en': naw.ContextualWordEmbsAug(
                model_path='xlm-roberta-base',
                action="substitute",
                aug_p=aug_p
            ),
            'ar': naw.ContextualWordEmbsAug(
                model_path='xlm-roberta-base',
                action="substitute",
                aug_p=aug_p
            )
        }
    
    def augment_text(self, text: str, lang: str) -> List[str]:
        try:
            if random.random() < self.aug_p:
                return self.augmenters[lang].augment(text, n=3)
            return [text]
        except Exception as e:
            logger.error(f"Augmentation failed: {str(e)}")
            return [text]

    def create_augmented_loader(self, df, tokenizer, lang="en", is_train=True):
        class AugmentedDataset(Dataset):
            def __init__(self, augmenter, df, lang):
                self.augmenter = augmenter
                self.lang = lang
                self.df = df
                self.samples = []
                
                for _, row in self.df.iterrows():
                    base_text = (
                        f"Major: {row[f'major_1_{self.lang}']} "
                        f"Job: {row[f'job_1_{self.lang}']} "
                        f"Hobby: {row[f'hobby_{self.lang}']}"
                    )
                    self.samples.extend(self.augmenter.augment_text(base_text, self.lang))
                    
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                encoding = tokenizer(
                    self.samples[idx],
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(self.df.iloc[idx//3]['label_code'])  # Handle augmented samples
                }

        dataset = AugmentedDataset(self, df, lang)
        return DataLoader(
            dataset,
            batch_size=4,
            shuffle=is_train,
            collate_fn=lambda batch: {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
                'labels': torch.stack([x['labels'] for x in batch])
            }
        )