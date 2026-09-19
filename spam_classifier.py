"""
Spam Classifier using Machine Learning
=======================================
Classifies SMS messages as "Spam" or "Ham" (not spam) using a
Multinomial Naive Bayes model with TF-IDF text vectorization.

Uses the real UCI SMS Spam Collection dataset (5,572 messages).
Accuracy on this dataset typically reaches ~98%.

Libraries: scikit-learn, pandas
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = "data/SMSSpamCollection"


def load_data():
    """Load the UCI SMS Spam Collection tab-separated dataset."""
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} messages")
    print(df["label"].value_counts())

    # Features (message text) and target (spam/ham)
    X = df["message"]
    y = df["label"]

    # Train/test split (stratified keeps class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Convert text to numeric features with TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    print("\n=== Model Evaluation ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Live predictions on new messages
    print("\n=== Live Predictions ===")
    test_messages = [
        "Congratulations! You have won a free iPhone. Click here to claim your prize now",
        "Hey, are we still meeting for lunch at 1pm?",
        "URGENT: Your account has been suspended. Call 0900 061 0146 to reactivate now",
        "Thanks for sending the report, I will review it tonight",
        "FREE MONEY!! Reply WIN to 87575 and get 2000 pounds credited instantly",
    ]
    for msg in test_messages:
        vec = vectorizer.transform([msg])
        prob = model.predict_proba(vec)[0]
        pred = "SPAM" if prob[1] > 0.5 else "HAM"
        print(f"{pred:>5}  ({max(prob):.2%} confidence)  {msg[:45]}")


if __name__ == "__main__":
    main()
