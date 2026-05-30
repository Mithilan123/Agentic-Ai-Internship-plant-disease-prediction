import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip()

df = df.drop_duplicates().reset_index(drop=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)      
    text = re.sub(r"\s+", " ", text).strip()        
    return text

def extract_text_features(text):
    text = clean_text(text)
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    word_count = len(words)
    unique_words = len(set(words))
    total_chars = sum(len(w) for w in words)

    vocab_richness = unique_words / word_count if word_count > 0 else 0
    avg_word_len = total_chars / word_count if word_count > 0 else 0
    sentence_count = len(sentences) if len(sentences) > 0 else 1
    text_length = len(text)

    return pd.Series({
        "vocab_richness": vocab_richness,
        "avg_word_len": avg_word_len,
        "sentence_count": sentence_count,
        "text_length": text_length
    })

if "text_" in df.columns:
    text_features = df["text_"].apply(extract_text_features)
    df["vocab_richness"] = text_features["vocab_richness"]
    df["avg_word_len"] = text_features["avg_word_len"]
    df["sentence_count"] = text_features["sentence_count"]
    df["text_length"] = text_features["text_length"]

if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

if "label_binary" in df.columns:
    y = df["label_binary"].astype(int)

# Otherwise create it from label
elif "label" in df.columns:
    # Based on your dataset screenshot:
    # CG = 0, OR = 1
    mapping = {
        "CG": 0,
        "OR": 1,
        "fake": 0,
        "genuine": 1,
        "real": 1
    }
    y = df["label"].map(mapping)

    if y.isnull().any():
        raise ValueError("Some labels could not be mapped. Check the 'label' column values.")

    y = y.astype(int)

else:
    raise ValueError("Target column not found. Need either 'label_binary' or 'label'.")

feature_cols = [
    "rating",
    "vocab_richness",
    "avg_word_len",
    "sentence_count",
    "text_length"
]

missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")

X = df[feature_cols].copy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


selector_estimator = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

selector = SelectFromModel(selector_estimator, threshold="median")
selector.fit(X_train_imputed, y_train)

X_train_selected = selector.transform(X_train_imputed)
X_test_selected = selector.transform(X_test_imputed)

selected_mask = selector.get_support()
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]

print("Selected Features:", selected_features)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_selected, y_train)

y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

artifacts = {
    "imputer": imputer,
    "selector": selector,
    "model": model,
    "selected_features": selected_features
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("Model saved as model.pkl")