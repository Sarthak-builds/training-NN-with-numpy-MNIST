# Training a Neural Network with NumPy on MNIST

A from-scratch 2-layer neural network built using only NumPy, trained on the Kaggle Digit Recognizer (MNIST) dataset.

## Structure

```
├── data/               # Dataset CSVs (not committed)
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── preprocess.py   # Data loading, normalization, train/test split
│   ├── model.py        # Neural network architecture and training functions
│   └── train.py        # Entry point — trains the model and evaluates accuracy
├── LOG.md              # Short description of every function and variable
├── requirements.txt
└── .gitignore
```

## Architecture

- **Input layer**: 784 neurons (28×28 flattened pixels)
- **Hidden layer**: 10 neurons with ReLU activation
- **Output layer**: 10 neurons (digits 0–9) with Softmax activation
- **Optimizer**: Gradient Descent

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Get the dataset from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data) and place `train.csv` and `test.csv` inside a `data/` folder at the repo root.

Or use the Kaggle CLI:
```bash
kaggle competitions download -c digit-recognizer -p data
```
Then unzip it into `data/`.

### 3. Train and evaluate
```bash
python src/train.py
```

## Output

Training prints accuracy every 50 epochs:
```
 Epoch:  0
 Accuracy:  0.1023
 Epoch:  50
 Accuracy:  0.6541
 ...
Test Set Accuracy: 0.8850
```

## Dataset

- **Source**: [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
- **Train samples**: 42,000 (90% train / 10% validation split)
- **Features**: 784 pixel values per image (grayscale, normalized 0–1)
- **Classes**: 10 (digits 0–9)
