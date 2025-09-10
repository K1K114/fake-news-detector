# evaluate.py
import argparse
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main(data_path: str, model_path: str, seed: int = 7) -> None:
    # Load data
    df = pd.read_csv(data_path).dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["text_clean"] = df["text"].astype(str).map(clean_text)
    X = df["text_clean"].values
    y = df["label"].values

    # Same split as train for fair holdout
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # Load model
    model = joblib.load(model_path)

    # Predict + report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["FAKE","REAL"]); ax.set_yticklabels(["FAKE","REAL"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=180)
    print("Saved -> confusion_matrix.png")

    # ROC AUC if probabilities are available (e.g., LogisticRegression)
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(X_test)
        # treat 'REAL' as positive class if present
        classes = list(model.named_steps["clf"].classes_)
        if "REAL" in classes:
            pos_idx = classes.index("REAL")
            y_true = (y_test == "REAL").astype(int)
            y_score = proba[:, pos_idx]
            auc = roc_auc_score(y_true, y_score)
            print(f"ROC AUC (REAL as positive): {auc:.4f}")

if __name__ == "__main__":
    import numpy as np  # only needed for cm annotations
    parser = argparse.ArgumentParser(description="Evaluate saved fake news model.")
    parser.add_argument("--data", default="news.csv", help="Path to CSV with columns: text,label")
    parser.add_argument("--model", default="fake_news_detector.joblib", help="Path to saved model")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args.data, args.model, args.seed)
