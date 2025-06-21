import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time

# Argument parser for model directory
parser = argparse.ArgumentParser(description='Test a saved fake news model.')
parser.add_argument('--model_dir', type=str, default='bert_fakenews_model', help='Path to the saved model directory')
args = parser.parse_args()

# Paths
DATA_DIR = './dataset'
TEST_FILE = os.path.join(DATA_DIR, 'test_preprocessed.csv')
MODEL_LOAD_PATH = args.model_dir

# Loading with progress bar and icons
print('\033[92m\u2714\ufe0f [INFO] Loading test dataset...\033[0m')
for _ in tqdm(range(100), desc='\U0001F4D6 Loading test set', bar_format='{l_bar}\033[92m{bar}\033[0m{r_bar}', ncols=70):
    time.sleep(0.005)
test_df = pd.read_csv(TEST_FILE)
print('\033[92m\u2714\ufe0f [INFO] Test dataset loaded!\033[0m')

# Only use statement and label_encoded
test_df = test_df[['statement', 'label_encoded']]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH)

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

test_dataset = NewsDataset(test_df, tokenizer)

# Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_LOAD_PATH)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    print('Classification Report:', classification_report(labels, preds))
    return {'accuracy': acc}

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=32,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Evaluate on test set
print('Evaluating on test set...')
test_results = trainer.evaluate(test_dataset)
print('Test set results:', test_results) 