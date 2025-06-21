import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Paths
DATA_DIR = './dataset'
TRAIN_FILE = os.path.join(DATA_DIR, 'train_preprocessed.csv')
VALID_FILE = os.path.join(DATA_DIR, 'valid_preprocessed.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_preprocessed.csv')
MODEL_NAME = 'bert-base-uncased'
MODEL_SAVE_PATH = './bert_fakenews_model'

# Loading with progress bar and icons
print('\033[92m\u2714\ufe0f [INFO] Loading datasets...\033[0m')
for desc, file in zip([
    '\U0001F4D6 Loading train set',
    '\U0001F4D6 Loading validation set',
    '\U0001F4D6 Loading test set'],
    [TRAIN_FILE, VALID_FILE, TEST_FILE]):
    for _ in tqdm(range(100), desc=desc, bar_format='{l_bar}\033[92m{bar}\033[0m{r_bar}', ncols=70):
        time.sleep(0.005)
    if 'train' in file:
        train_df = pd.read_csv(file)
    elif 'valid' in file:
        valid_df = pd.read_csv(file)
    elif 'test' in file:
        test_df = pd.read_csv(file)
print('\033[92m\u2714\ufe0f [INFO] All datasets loaded!\033[0m')

# Only use statement and label_encoded
train_df = train_df[['statement', 'label_encoded']]
valid_df = valid_df[['statement', 'label_encoded']]
test_df = test_df[['statement', 'label_encoded']]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['statement'].tolist()
        self.labels = df['label_encoded'].tolist()
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True, padding=False, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in item.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = NewsDataset(train_df, tokenizer)
valid_dataset = NewsDataset(valid_df, tokenizer)
test_dataset = NewsDataset(test_df, tokenizer)

# Model
num_labels = len(train_df['label_encoded'].unique())
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    print('Classification Report:', classification_report(labels, preds))
    return {'accuracy': acc}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate on test set
print('Evaluating on test set...')
test_results = trainer.evaluate(test_dataset)
print('Test set results:', test_results)

# Save model
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f'Model saved to {MODEL_SAVE_PATH}') 