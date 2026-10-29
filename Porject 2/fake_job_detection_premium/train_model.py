"""Train the text classifier from a labeled CSV dataset.

Expected CSV label column: fraudulent (0 = legitimate, 1 = fraudulent).
Recommended source: a reputable Real/Fake Job Posting dataset.
The included data/sample_jobs.csv is synthetic demo data only and is NOT suitable
for real-world accuracy claims or production decisions.
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "data" / "sample_jobs.csv"
MODEL_PATH = BASE_DIR / "models" / "job_risk_model.joblib"

TEXT_COLUMNS = [
    "title", "company_profile", "description", "requirements",
    "benefits", "employment_type", "location"
]

def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Dataset not found: {DATA_PATH}\n"
            "Place a labeled CSV there or run: python train_model.py path/to/your_dataset.csv"
        )
    df = pd.read_csv(DATA_PATH)
    if "fraudulent" not in df.columns:
        raise SystemExit("CSV must contain a 'fraudulent' column (0=legitimate, 1=fraudulent).")
    present = [c for c in TEXT_COLUMNS if c in df.columns]
    if not present:
        raise SystemExit("CSV needs at least one text column such as title or description.")

    df = df.dropna(subset=["fraudulent"]).copy()
    df["fraudulent"] = pd.to_numeric(df["fraudulent"], errors="coerce")
    df = df[df["fraudulent"].isin([0, 1])]
    df["fraudulent"] = df["fraudulent"].astype(int)
    df["combined_text"] = df[present].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    df = df[df["combined_text"].str.len() > 0]
    if df["fraudulent"].nunique() != 2 or len(df) < 10:
        raise SystemExit("Need at least 10 non-empty labeled rows and examples from both classes.")

    X, y = df["combined_text"], df["fraudulent"]
    # Stratify only when each class has enough examples for a split.
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), max_features=80000,
            sublinear_tf=True, min_df=1
        )),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Dataset: {DATA_PATH}")
    print(f"Rows used: {len(df)} | Training: {len(X_train)} | Test: {len(X_test)}")
    print("\nEvaluation on held-out test split:")
    print(classification_report(y_test, pred, labels=[0, 1], target_names=["legitimate", "fraudulent"], zero_division=0))
    print("Confusion matrix [legitimate, fraudulent]:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")
    if DATA_PATH.name == "sample_jobs.csv":
        print("WARNING: This is synthetic demo data. Metrics do not represent real-world performance.")

if __name__ == "__main__":
    main()
