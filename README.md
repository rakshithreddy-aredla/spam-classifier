# Spam Classifier 📧

A machine learning model that classifies SMS messages as **Spam** or **Ham** (not spam), achieving **~97% accuracy** on the real UCI SMS Spam Collection dataset (5,572 messages).

## 🎯 The Problem

Spam filtering is a classic real-world ML problem. Every message must be quickly and accurately classified as spam or legitimate, with very few false positives (important emails being deleted).

## 🔧 How it works

| Step | What happens |
|------|-------------|
| 1. Data | UCI SMS Spam Collection - 4,825 ham + 747 spam messages |
| 2. Preprocessing | Lowercase, remove stop words, TF-IDF vectorization converts text to numbers |
| 3. Model | **Multinomial Naive Bayes** - uses Bayes' theorem to compute P(spam \| words) |
| 4. Evaluation | Accuracy, precision, recall, F1, confusion matrix |

## 🧠 Why Naive Bayes?

Naive Bayes is perfect for text classification because:
- It handles high-dimensional sparse text data well
- It's extremely fast to train and predict
- It works surprisingly well on classification despite its "naive" independence assumption

## 🚀 How to run

```bash
pip install -r requirements.txt
python spam_classifier.py
```

## 📊 Results

```
Accuracy: 0.9706

              precision    recall  f1-score   support
         ham       0.97      1.00      0.98      1206
        spam       1.00      0.78      0.88       187
```

**Key insight:** The model catches 100% of ham messages (zero false positives) and 78% of spam. If you needed to catch more spam, you could lower the decision threshold — a classic precision/recall tradeoff.

## 🏗️ Project Structure

```
01-spam-classifier/
├── spam_classifier.py     # Main script
├── data/SMSSpamCollection # UCI dataset
├── requirements.txt
└── README.md
```

## 📚 ML Concepts Covered

- Train/test splitting (with stratification)
- Text preprocessing & TF-IDF vectorization
- Naive Bayes probability theory
- Classification metrics: accuracy, precision, recall, F1
- Precision/recall tradeoff
- Confusion matrix interpretation
