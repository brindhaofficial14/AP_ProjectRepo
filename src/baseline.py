# src/baseline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump, load
import argparse, os

def train_baseline(csv_path: str, model_dir: str = "models/baseline"):
    df = pd.read_csv(csv_path)
    X, y = df["text"].astype(str), df["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipe.fit(X_tr, y_tr)
    os.makedirs(model_dir, exist_ok=True)
    dump(pipe, os.path.join(model_dir, "tfidf.joblib"))
    print("Saved baseline to", model_dir)
    print(classification_report(y_te, pipe.predict(X_te), digits=3))

def predict_baseline(text: str, model_dir: str = "models/baseline"):
    pipe = load(os.path.join(model_dir, "tfidf.joblib"))
    proba = pipe.predict_proba([text])[0][1]
    label = int(proba >= 0.5)
    return {"label": label, "score": float(proba)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    args = ap.parse_args()
    train_baseline(args.train_csv)
