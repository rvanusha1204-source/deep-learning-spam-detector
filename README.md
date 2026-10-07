# 📩 Spam Email/SMS Detection using Deep Learning (Bi-LSTM)

A deep learning system that classifies text messages as **Spam** or **Not Spam (Ham)** by analysing their textual content. Built with **PyTorch**, compared against a classical **TF-IDF + Naive Bayes** baseline, and shipped with a **Streamlit** web app for live predictions.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

## Problem Statement
Develop a Deep Learning-based Spam Email/SMS Detection System that automatically classifies incoming text messages as Spam or Not Spam (Ham) by analysing their textual content.

## Features
- Text cleaning that normalises URLs, long phone numbers and currency amounts into tokens
- Bi-LSTM classifier with embedding, dropout, class weighting for imbalanced data and early stopping
- Baseline model (TF-IDF + Naive Bayes) for comparison
- Evaluation with accuracy, precision, recall, F1 and a confusion matrix
- Command-line prediction and an interactive Streamlit web app

## Model Architecture
```
Input text -> Cleaning -> Tokenisation (10k vocab, length 60)
           -> Embedding (64) -> Dropout -> Bidirectional LSTM (64)
           -> Masked mean pooling -> Dropout -> Dense (32, ReLU) -> Sigmoid -> Spam / Ham
```

## Dataset
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): 5,572 messages (about 13% spam). `train.py` downloads it automatically to `data/sms.tsv`.
To extend the system to emails, append rows in `label<TAB>text` format (`ham` / `spam`) from an email corpus such as Enron-Spam or SpamAssassin, and consider increasing `MAX_LEN` in `common.py`.

## Results
Evaluated on a held-out test set of 1,115 messages (966 Ham, 149 Spam).

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|-------|----------|----------------|-------------|---------|
| TF-IDF + Naive Bayes | 97.13% | 1.00 | 0.79 | 0.88 |
| Bi-LSTM (PyTorch) | 98.12% | 0.94 | 0.92 | 0.93 |

![Confusion Matrix](outputs/confusion_matrix.png)
![Training Curves](outputs/training_curves.png)

**Key findings**
- The Bi-LSTM beats the baseline in accuracy (98.12% vs 97.13%) and in spam F1 (0.93 vs 0.88).
- Naive Bayes never flags a genuine message as spam (precision 1.00), but it misses about 21% of real spam (recall 0.79).
- The Bi-LSTM catches far more spam (recall 0.92) at the cost of a few more false alarms (precision 0.94).
- The dataset is imbalanced (4,825 Ham vs 747 Spam). Class weighting and early stopping were used to handle this and to avoid overfitting.

## Project Structure
```
.
├── common.py          # text cleaning, encoding, Bi-LSTM model
├── train.py           # data loading, training, evaluation, saving
├── predict.py         # command-line inference
├── app.py             # Streamlit web app
├── requirements.txt
├── data/              # dataset (auto-downloaded)
├── model/             # saved weights and vocabulary
└── outputs/           # confusion matrix and training curves
```

## Getting Started
```bash
# 1. Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create and activate a virtual environment (Windows)
py -3.11 -m venv venv
venv\Scripts\activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install scikit-learn pandas numpy matplotlib seaborn streamlit

# 4. Train
python train.py

# 5. Predict
python predict.py "Congratulations! You won a free prize, call now"

# 6. Launch the web app
python -m streamlit run app.py
```
On macOS/Linux, activate with `source venv/bin/activate`.

## Example
```
$ python predict.py "Congratulations! You won a free prize. Call 09061701461 now to claim."
Spam  (spam probability 0.99x)
```

## Future Improvements
- Pre-trained embeddings (GloVe) or a fine-tuned DistilBERT model
- Train on a combined email + SMS dataset
- Compare against CNN and GRU models
- Tune the decision threshold to reduce false positives


