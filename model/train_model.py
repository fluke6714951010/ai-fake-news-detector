```python
# model/train_model.py

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.preprocess import clean_text


print("=================================")
print("     AI Fake News Training")
print("=================================")


# -------------------------------
# 1. Load Dataset
# -------------------------------
DATA_PATH = "data/news_dataset.csv"

if not os.path.exists(DATA_PATH):
    print("❌ Dataset not found:", DATA_PATH)
    exit()

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

# ป้องกัน error ถ้ามีค่า NaN
data["text"] = data["text"].astype(str)

# -------------------------------
# 2. Clean Text
# -------------------------------
print("Cleaning text...")

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

# -------------------------------
# 3. Vectorize Text
# -------------------------------
print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1,2)
)

X_vec = vectorizer.fit_transform(X)

# -------------------------------
# 4. Split Dataset
# -------------------------------
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 5. Train Model
# -------------------------------
print("Training model...")

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# -------------------------------
# 6. Test Model
# -------------------------------
print("Testing model...")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Model Accuracy:", round(acc*100,2), "%")

# -------------------------------
# 7. Save Model
# -------------------------------
print("Saving model...")

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model saved successfully!")
print("=================================")
print("Training Completed")
print("=================================")
```
