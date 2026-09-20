import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD ENHANCED DATASET
# ============================================================

df = pd.read_csv("enhanced_training_data.csv")

X = df["text"]
y = df["intent"]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. BUILD ML PIPELINE
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ============================================================
# 4. TRAIN
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 5. PREDICT
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 6. EVALUATE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("SHOPSMART RETAIL - ENHANCED TEXT CLASSIFIER")
print("=" * 60)

print(f"Total samples:     {len(df)}")
print(f"Training samples:  {len(X_train)}")
print(f"Testing samples:   {len(X_test)}")
print(f"Number of intents: {y.nunique()}")
print()

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 7. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "shopsmart_intent_classifier.joblib"
)

print()
print("Model saved as:")
print("shopsmart_intent_classifier.joblib")