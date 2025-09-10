import argparse
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------
# Text cleaning
# -----------------------
def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)      # remove URLs
    s = re.sub(r"[^a-z0-9\s]", " ", s)           # keep alphanum only
    s = re.sub(r"\s+", " ", s).strip()
    return s



# -----------------------
# Training routine
# -----------------------
def train(data_path: str, model_out: str, random_state: int = 7) -> None:
    # 1) Load
    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset=["text"]).dropna(subset=["text", "label"])
    df["text_clean"] = df["text"].astype(str).map(clean_text)
    X = df["text_clean"].values
    y = df["label"].values

    # 2) Split (stratified)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 3) Shared TF-IDF (fixed, strong defaults)
    tfidf = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        max_features=20000,
    )

    # 4) Two comparable pipelines
    lr_pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])
    pac_pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", PassiveAggressiveClassifier(max_iter=2000)),
    ])

    # 5) Quick CV to choose winner
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    lr_cv  = cross_val_score(lr_pipe,  x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
    pac_cv = cross_val_score(pac_pipe, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
    print(f"CV f1_macro -> LogisticRegression: {lr_cv:.4f} | PassiveAggressive: {pac_cv:.4f}")

    winner = lr_pipe if lr_cv >= pac_cv else pac_pipe
    winner_name = "LogisticRegression" if winner is lr_pipe else "PassiveAggressiveClassifier"

    # 6) Fit winner on full train
    winner.fit(x_train, y_train)

    # 7) Holdout evaluation
    y_pred = winner.predict(x_test)
    print(f"\nChosen model: {winner_name}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix (rows=true, cols=pred):\n",
          confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"]))

    # 8) Persist pipeline (vectorizer + model)
    joblib.dump(winner, model_out)
    print(f"\nSaved model pipeline -> {model_out}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detector")
    parser.add_argument("--data", default="news.csv", help="Path to CSV with columns: text,label")
    parser.add_argument("--out",  default="fake_news_detector.joblib", help="Output path for saved model")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    train(data_path=args.data, model_out=args.out, random_state=args.seed)
