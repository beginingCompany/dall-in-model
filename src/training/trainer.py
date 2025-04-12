import torch
import logging
from transformers import XLMRobertaModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from config.paths import PathConfig
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW

logger = logging.getLogger(__name__)
paths = PathConfig()

class PersonalityDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = self.data.iloc[idx]['label_code']
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PersonalityClassifierTrainer:
    def __init__(self, num_labels: int):
        # Ensure all parameters are defined
        self.batch_size = 16
        self.epochs = 60
        self.learning_rate = 1e-5
        self.grad_accum_steps = 2
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_loader(self, df, tokenizer, is_train=True):
        return DataLoader(
            PersonalityDataset(df, tokenizer),
            batch_size=self.batch_size,
            shuffle=is_train,
            num_workers=2
        )

    def initialize_model(self):
        class CustomModel(nn.Module):
            def __init__(self, num_labels):
                super().__init__()
                self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(768, num_labels)
                self.loss_fct = nn.CrossEntropyLoss()

                # Freeze first 8 layers
                for layer in self.roberta.encoder.layer[:8]:
                    for param in layer.parameters():
                        param.requires_grad = False

            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                pooled_output = self.dropout(outputs.pooler_output)
                logits = self.classifier(pooled_output)
                loss = self.loss_fct(logits, labels) if labels is not None else None
                return {'logits': logits, 'loss': loss}

        return CustomModel(self.num_labels).to(self.device)

    def create_optimizer_scheduler(self, model, num_training_steps):
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps//10,
            num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _validate_loader(self, loader, name: str):
        """Data loader validation method"""
        try:
            batch = next(iter(loader))
            logger.info(f"{name} loader contains {len(loader.dataset)} samples")
            logger.info(f"Sample batch shape: {batch['input_ids'].shape}")
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise

    def train(self, train_df, val_df, tokenizer):
        model = self.initialize_model()
        train_loader = self._create_loader(train_df, tokenizer)
        val_loader = self._create_loader(val_df, tokenizer, is_train=False)
        
        # Calculate training steps
        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        optimizer, scheduler = self.create_optimizer_scheduler(model, total_steps)

        # Data validation
        self._validate_loader(train_loader, "Training")
        self._validate_loader(val_loader, "Validation")

        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler, epoch)
            val_metrics = self.validate(model, val_loader)

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}\n"
                f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%\n"
                f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%"
            )

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(model.state_dict(), paths.MODEL_PATH.as_posix())

        return model

    def train_epoch(self, model, loader, optimizer, scheduler, epoch: int):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            loss = outputs['loss'] / self.grad_accum_steps
            loss.backward()

            if (i + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Metrics
            preds = torch.argmax(outputs['logits'], dim=1)
            correct += (preds == inputs["labels"]).sum().item()
            total_samples += inputs["labels"].size(0)
            total_loss += outputs['loss'].item()

            progress_bar.set_postfix({
                'loss': total_loss / (i + 1),
                'acc': f"{100 * correct / total_samples:.2f}%",
                'lr': scheduler.get_last_lr()[0]
            })

        return {
            "loss": total_loss / len(loader),
            "accuracy": 100 * correct / total_samples
        }

    def validate(self, model, loader):
        model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**inputs)
                total_loss += outputs['loss'].item()
                preds = torch.argmax(outputs['logits'], dim=1)
                correct += (preds == inputs["labels"]).sum().item()
                total_samples += inputs["labels"].size(0)
                
        return {
            "loss": total_loss / len(loader),
            "accuracy": 100 * correct / total_samples
        }
