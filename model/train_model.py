# model/train_model.py

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from utils.preprocess import clean_text

print("=== AI Fake News Training ===")

DATA_PATH = "data/news_dataset.csv"

if not os.path.exists(DATA_PATH):
    print("Dataset not found")
    exit()

data = pd.read_csv(DATA_PATH)

data["text"] = data["text"].astype(str)

print("Cleaning text...")
data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

print("Vectorizing...")

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,3),
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

model = LinearSVC()

model.fit(X_train, y_train)

print("Testing model...")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

print("Saving model...")

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Training Complete")
