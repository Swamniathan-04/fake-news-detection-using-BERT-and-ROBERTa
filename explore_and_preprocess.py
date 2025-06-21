import pandas as pd
import numpy as np
import os

# File paths
DATA_DIR = './dataset'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
VALID_CSV = os.path.join(DATA_DIR, 'valid.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

# Column names as per README
COLUMNS = [
    'id', 'label', 'statement', 'subjects', 'speaker', 'job_title', 'state', 'party',
    'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts',
    'context'
]

def load_data(path):
    return pd.read_csv(path, names=COLUMNS, header=None)

def explore_data(df, name):
    print(f'--- {name} ---')
    print('Shape:', df.shape)
    print('Sample rows:')
    print(df.head())
    print('\nMissing values per column:')
    print(df.isnull().sum())
    print('\nLabel distribution:')
    print(df['label'].value_counts())
    print('\n')

def clean_text(text):
    if pd.isnull(text):
        return ''
    return str(text).strip().lower()

def preprocess(df):
    df['statement'] = df['statement'].apply(clean_text)
    df['label'] = df['label'].str.strip().str.lower()
    return df

def encode_labels(df, label_map=None):
    if label_map is None:
        unique_labels = sorted(df['label'].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_encoded'] = df['label'].map(label_map)
    return df, label_map

def main():
    # Load data
    train_df = load_data(TRAIN_CSV)
    valid_df = load_data(VALID_CSV)
    test_df = load_data(TEST_CSV)

    # Explore data
    explore_data(train_df, 'Train')
    explore_data(valid_df, 'Validation')
    explore_data(test_df, 'Test')

    # Preprocess
    train_df = preprocess(train_df)
    valid_df = preprocess(valid_df)
    test_df = preprocess(test_df)

    # Encode labels (fit on train, apply to all)
    train_df, label_map = encode_labels(train_df)
    valid_df, _ = encode_labels(valid_df, label_map)
    test_df, _ = encode_labels(test_df, label_map)

    # Save preprocessed files
    train_df.to_csv(os.path.join(DATA_DIR, 'train_preprocessed.csv'), index=False)
    valid_df.to_csv(os.path.join(DATA_DIR, 'valid_preprocessed.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test_preprocessed.csv'), index=False)
    print('Preprocessing complete. Preprocessed files saved.')

if __name__ == '__main__':
    main() 