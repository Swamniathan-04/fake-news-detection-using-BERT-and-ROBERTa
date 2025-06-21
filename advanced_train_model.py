import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, DataCollatorWithPadding,
    RobertaTokenizer, RobertaForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import random
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Paths
DATA_DIR = './dataset'
TRAIN_FILE = os.path.join(DATA_DIR, 'train_preprocessed.csv')
VALID_FILE = os.path.join(DATA_DIR, 'valid_preprocessed.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_preprocessed.csv')
MODEL_SAVE_PATH = './advanced_bert_fakenews_model'

# Hyperparameters for achieving high accuracy
BATCH_SIZE = 8  # Smaller batch size for better gradient updates
LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
EPOCHS = 10  # More epochs
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4

print('\033[92m\u2714\ufe0f [INFO] Loading datasets...\033[0m')
train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)
test_df = pd.read_csv(TEST_FILE)
print('\033[92m\u2714\ufe0f [INFO] All datasets loaded!\033[0m')

# Data augmentation functions
def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def augment_text(text, augmentation_factor=0.3):
    """Augment text by replacing words with synonyms"""
    words = word_tokenize(text)
    augmented_words = []
    
    for word in words:
        if random.random() < augmentation_factor and len(word) > 3:
            synonyms = get_synonyms(word)
            if synonyms:
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    
    return ' '.join(augmented_words)

def create_augmented_dataset(df, augmentation_factor=0.3, num_augmentations=2):
    """Create augmented dataset"""
    augmented_data = []
    
    for _, row in tqdm(df.iterrows(), desc="Creating augmented data", total=len(df)):
        # Original data
        augmented_data.append({
            'statement': row['statement'],
            'label_encoded': row['label_encoded']
        })
        
        # Augmented data
        for _ in range(num_augmentations):
            augmented_text = augment_text(row['statement'], augmentation_factor)
            augmented_data.append({
                'statement': augmented_text,
                'label_encoded': row['label_encoded']
            })
    
    return pd.DataFrame(augmented_data)

# Create augmented training data
print('\033[92m\u2714\ufe0f [INFO] Creating augmented training data...\033[0m')
augmented_train_df = create_augmented_dataset(train_df, augmentation_factor=0.2, num_augmentations=3)
print(f'Original training samples: {len(train_df)}')
print(f'Augmented training samples: {len(augmented_train_df)}')

# Advanced Dataset class with multiple features
class AdvancedNewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df['statement'].tolist()
        self.labels = df['label_encoded'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add additional features
        self.speakers = df.get('speaker', ['unknown'] * len(df)).tolist()
        self.subjects = df.get('subjects', ['unknown'] * len(df)).tolist()
        self.contexts = df.get('context', ['unknown'] * len(df)).tolist()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        speaker = self.speakers[idx]
        subject = self.subjects[idx]
        context = self.contexts[idx]
        
        # Create enhanced input with additional context
        enhanced_text = f"Speaker: {speaker} | Subject: {subject} | Context: {context} | Statement: {text}"
        
        item = self.tokenizer(
            enhanced_text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in item.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Advanced model with custom head
class AdvancedFakeNewsModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.3):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        # Add custom layers for better performance
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# Ensemble model
class EnsembleModel:
    def __init__(self, models, tokenizers):
        self.models = models
        self.tokenizers = tokenizers
        
    def predict(self, text):
        predictions = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
        
        # Return most common prediction
        return max(set(predictions), key=predictions.count)

# Training function with advanced techniques
def train_advanced_model(model_name, train_dataset, valid_dataset, num_labels):
    print(f'\033[92m\u2714\ufe0f [INFO] Training with {model_name}...\033[0m')
    
    # Initialize model
    if 'roberta' in model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Advanced training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name.replace("/", "_")}',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f'./logs_{model_name.replace("/", "_")}',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_steps=WARMUP_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=True,  # Use mixed precision
        dataloader_pin_memory=False,
        report_to='none',
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate on training data
    train_results = trainer.evaluate(train_dataset)
    print(f'Training accuracy: {train_results["eval_accuracy"]:.4f}')
    
    return model, tokenizer, trainer

# Main training pipeline
def main():
    print('\033[92m\u2714\ufe0f [INFO] Starting advanced training pipeline...\033[0m')
    
    # Prepare datasets
    num_labels = len(augmented_train_df['label_encoded'].unique())
    
    # Model configurations
    models_to_train = [
        'roberta-base',
        'roberta-large',
        'bert-base-uncased',
        'bert-large-uncased'
    ]
    
    trained_models = []
    trained_tokenizers = []
    
    # Train multiple models
    for model_name in models_to_train:
        try:
            print(f'\n\033[94m[INFO] Training {model_name}...\033[0m')
            
            # Create datasets
            if 'roberta' in model_name.lower():
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            train_dataset = AdvancedNewsDataset(augmented_train_df, tokenizer)
            valid_dataset = AdvancedNewsDataset(valid_df, tokenizer)
            
            # Train model
            model, tokenizer, trainer = train_advanced_model(model_name, train_dataset, valid_dataset, num_labels)
            
            trained_models.append(model)
            trained_tokenizers.append(tokenizer)
            
            # Save individual model
            model_save_path = f'./{model_name.replace("/", "_")}_model'
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f'Model saved to {model_save_path}')
            
        except Exception as e:
            print(f'Error training {model_name}: {e}')
            continue
    
    # Create ensemble
    if len(trained_models) > 1:
        print('\n\033[94m[INFO] Creating ensemble model...\033[0m')
        ensemble = EnsembleModel(trained_models, trained_tokenizers)
        
        # Test ensemble on training data
        correct = 0
        total = 0
        
        for _, row in tqdm(augmented_train_df.iterrows(), desc="Testing ensemble", total=len(augmented_train_df)):
            pred = ensemble.predict(row['statement'])
            if pred == row['label_encoded']:
                correct += 1
            total += 1
        
        ensemble_accuracy = correct / total
        print(f'Ensemble training accuracy: {ensemble_accuracy:.4f}')
    
    print('\n\033[92m\u2714\ufe0f [INFO] Advanced training completed!\033[0m')

if __name__ == "__main__":
    main() 