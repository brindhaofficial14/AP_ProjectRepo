# src/baseline.py
import pandas as pd  # data loading/manipulation
from sklearn.model_selection import train_test_split  # stratified split
from sklearn.pipeline import Pipeline  # glue vectorizer + classifier
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF features
from sklearn.linear_model import LogisticRegression  # linear baseline
from sklearn.metrics import classification_report  # summary metrics
from joblib import dump, load  # save/load model pipeline
import argparse, os  # CLI + filesystem

def train_baseline(csv_path: str, model_dir: str = "models/baseline"):
    """Train TF-IDF + Logistic Regression baseline and save to disk."""
    df = pd.read_csv(csv_path)  # load dataset
    X, y = df["text"].astype(str), df["label"].astype(int)  # features/labels
    # Stratified train/test split (20% test)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Define pipeline: character/word n-grams via TF-IDF + LR classifier
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)),  # unigrams/bigrams
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))    # handle class imbalance
    ])
    pipe.fit(X_tr, y_tr)  # train pipeline
    os.makedirs(model_dir, exist_ok=True)  # ensure output dir
    dump(pipe, os.path.join(model_dir, "tfidf.joblib"))  # persist model
    print("Saved baseline to", model_dir)  # info
    print(classification_report(y_te, pipe.predict(X_te), digits=3, zero_division=0))  # quick report
    

def predict_baseline(text: str, model_dir: str = "models/baseline"):
    """Load saved baseline and predict risk score/label for a single text."""
    pipe = load(os.path.join(model_dir, "tfidf.joblib"))  # load pipeline
    proba = pipe.predict_proba([text])[0][1]  # probability of unsafe class (1)
    label = int(proba >= 0.5)  # threshold at 0.5
    return {"label": label, "score": float(proba)}  # dict for consistency

if __name__ == "__main__":
    # Simple CLI to train the baseline
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)  # path to training CSV
    args = ap.parse_args()
    train_baseline(args.train_csv)  # train and print report
