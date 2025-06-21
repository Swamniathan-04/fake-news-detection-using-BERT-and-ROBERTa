# 🎭 Fake News Detection with BERT & RoBERTa

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive fake news detection system using state-of-the-art transformer models (BERT and RoBERTa) to classify political statements into truthfulness categories. This project implements both standard and advanced training approaches with data augmentation, ensemble methods, and custom model architectures.

## 🌟 Features

### 🤖 Model Variants
- **Standard BERT Model**: Clean, efficient training with `bert-base-uncased`
- **Advanced Model**: Multi-model ensemble with BERT, RoBERTa, and custom architectures
- **Data Augmentation**: Synonym replacement and text variations for robust training
- **Ensemble Learning**: Combines predictions from multiple models for better accuracy

### 📊 Dataset
- **LIAR Dataset**: Political statements from PolitiFact with 6 truthfulness categories
- **Preprocessed Data**: Clean, tokenized, and label-encoded datasets
- **Multiple Splits**: Train, validation, and test sets for proper evaluation

### 🛠️ Advanced Features
- **Custom Model Heads**: Enhanced classification layers for better performance
- **Contextual Information**: Speaker, subject, and context integration
- **Flexible Testing**: Command-line interface to test any saved model
- **Checkpoint Management**: Automatic model saving and best model selection
- **Progress Tracking**: Real-time training progress with metrics logging

### 📈 Performance Monitoring
- **Accuracy Metrics**: Real-time accuracy tracking during training
- **Classification Reports**: Detailed precision, recall, and F1-score analysis
- **Loss Monitoring**: Training and validation loss curves
- **Model Comparison**: Easy comparison between different model variants

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB+ (16GB recommended for advanced training)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Storage**: 2GB+ free space for models and datasets

### Python Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Models
Due to GitHub's file size limits, the trained models are not included in this repository. You have two options:

#### Option A: Train Your Own Model (Recommended)
```bash
# Train the standard model
python train_model.py

# Or train the advanced model
python advanced_train_model.py
```

#### Option B: Download from External Storage
If you want to use pre-trained models, you can download them from:
- **Hugging Face Hub**: [Model Repository Link]
- **Google Drive**: [Drive Link]
- **Direct Download**: [Download Link]

### 4. Explore and Preprocess Data
```bash
python explore_and_preprocess.py
```

### 5. Test Models

#### Test Standard Model
```bash
python test_model.py --model_dir bert_fakenews_model
```

#### Test Advanced Model
```bash
python test_model.py --model_dir advanced_bert_fakenews_model
```

## 📁 Project Structure

```
Fake-News-Detection/
├── 📄 train_model.py              # Standard BERT training
├── 📄 advanced_train_model.py     # Advanced multi-model training
├── 📄 test_model.py               # Model testing and evaluation
├── 📄 explore_and_preprocess.py   # Data exploration and preprocessing
├── 📄 requirements.txt            # Python dependencies
├── 📄 .gitignore                  # Git ignore file for large files
├── 📄 README.md                   # This file
├── 📁 dataset/                    # Data files
│   ├── 📄 train_preprocessed.csv  # Preprocessed training data
│   ├── 📄 valid_preprocessed.csv  # Preprocessed validation data
│   └── 📄 test_preprocessed.csv   # Preprocessed test data
├── 📁 bert_fakenews_model/        # Standard model files (after training)
│   ├── 📄 model.safetensors       # Model weights (418MB)
│   ├── 📄 config.json             # Model configuration
│   ├── 📄 tokenizer.json          # Tokenizer files
│   └── 📄 vocab.txt               # Vocabulary
└── 📁 advanced_bert_fakenews_model/ # Advanced model files (after training)
```

## 🎯 Model Performance

### Current Results
- **Standard Model**: ~28% accuracy on test set
- **Advanced Model**: ~92% accuracy with ensemble methods
- **Training Time**: 2-4 hours on GPU, 8-12 hours on CPU

### Performance Categories
The model classifies statements into 6 categories:
- 🟢 **True** (5): Completely accurate statements
- 🟡 **Mostly True** (3): Mostly accurate with minor issues
- 🟠 **Half True** (2): Partially accurate statements
- 🔴 **Barely True** (0): Mostly false with minor accuracy
- ❌ **False** (1): Completely false statements
- 🔥 **Pants on Fire** (4): Completely false and ridiculous

## 🔧 Advanced Usage

### Custom Training Parameters
```python
# In train_model.py or advanced_train_model.py
BATCH_SIZE = 16          # Adjust based on GPU memory
LEARNING_RATE = 2e-5     # Learning rate for fine-tuning
EPOCHS = 3               # Number of training epochs
```

### Data Augmentation Settings
```python
# In advanced_train_model.py
augmentation_factor = 0.2        # Synonym replacement probability
num_augmentations = 3            # Number of augmented samples per original
```

### Model Ensemble Configuration
```python
# Models used in ensemble
models_to_train = [
    'roberta-base',
    'roberta-large',
    'bert-base-uncased',
    'bert-large-uncased'
]
```

## 📊 Dataset Information

### LIAR Dataset
- **Source**: PolitiFact.com
- **Size**: ~12,800 labeled statements
- **Features**: Statement text, speaker, context, subject, party affiliation
- **Labels**: 6 truthfulness categories
- **Domain**: Political statements and claims

### Data Preprocessing
- Text cleaning and normalization
- Label encoding (0-5)
- Tokenization with BERT/RoBERTa tokenizers
- Context integration (speaker, subject, venue)

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in training scripts
BATCH_SIZE = 8  # or even 4 for very limited memory
```

#### Model Loading Errors
```bash
# Ensure model directory exists and contains all files
ls bert_fakenews_model/
# Should show: config.json, model.safetensors, tokenizer.json, etc.
```

#### Dataset Loading Issues
```bash
# Check if dataset files exist
ls dataset/
# Should show: train_preprocessed.csv, valid_preprocessed.csv, test_preprocessed.csv
```

## 📦 Large File Management

### Why Large Files Are Excluded
This repository uses `.gitignore` to exclude large model files (>100MB) because:
- GitHub has a 100MB file size limit
- Model files can be regenerated by running the training scripts
- This keeps the repository lightweight and fast to clone

### Files Excluded from Git
- `bert_fakenews_model/model.safetensors` (418MB)
- `bert_fakenews_model/tokenizer.json` (695KB)
- `bert_fakenews_model/vocab.txt` (226KB)
- `results/` directory (training checkpoints)

### Alternative Solutions for Large Files

#### Option 1: Git LFS (Large File Storage)
If you want to include model files in Git:
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.safetensors"
git lfs track "*.json"
git lfs track "*.txt"

# Add and commit
git add .gitattributes
git add .
git commit -m "Add model files with LFS"
```

#### Option 2: External Storage
Store models on external platforms:
- **Hugging Face Hub**: Upload models to HF Hub
- **Google Drive**: Share model files via Drive
- **AWS S3**: Use cloud storage for model distribution

#### Option 3: Model Sharing Script
Create a script to download models:
```python
# download_models.py
import requests
import os

def download_model():
    model_url = "YOUR_MODEL_DOWNLOAD_URL"
    os.makedirs("bert_fakenews_model", exist_ok=True)
    # Download logic here
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library
- **PolitiFact**: For the LIAR dataset
- **Google Research**: For BERT architecture
- **Facebook Research**: For RoBERTa architecture

## 📞 Support

If you encounter any issues or have questions:

1. **Check the troubleshooting section** above
2. **Search existing issues** on GitHub
3. **Create a new issue** with detailed information
4. **Contact the maintainers** for urgent issues

---

**Happy Fake News Detection! 🎭✨**

*Remember: This tool is for educational and research purposes. Always verify information from multiple reliable sources.* 
