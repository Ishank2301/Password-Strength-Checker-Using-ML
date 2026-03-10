import joblib
import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


LR_PIPELINE_PATH = "models/lr_password_pipeline.pkl"
XGB_PIPELINE_PATH = "models/xgb_password_pipeline.pkl"


def load_data():

    print("[train] Loading dataset...")

    df = pd.read_csv("data/processed/password_features.csv")

    X = df["password"]
    y = df["strength"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def create_lr_pipeline():

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=5000),
            ),
            ("model", LogisticRegression(solver="saga", max_iter=300, n_jobs=-1)),
        ]
    )


def create_xgb_pipeline():

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=5000),
            ),
            (
                "model",
                XGBClassifier(
                    tree_method="hist",
                    n_estimators=80,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"\nAccuracy: {acc:.4f}\n")

    print(classification_report(y_test, preds))


def save_model(model, path):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, path)

    print(f"[train] Model saved → {path}")


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data()

    print("\n[train] Training Logistic Regression Pipeline...")
    lr_pipeline = create_lr_pipeline()
    lr_pipeline.fit(X_train, y_train)

    print("\n--- Logistic Regression Results ---")
    evaluate_model(lr_pipeline, X_test, y_test)

    save_model(lr_pipeline, LR_PIPELINE_PATH)

    print("\n[train] Training XGBoost Pipeline...")
    xgb_pipeline = create_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    print("\n--- XGBoost Results ---")
    evaluate_model(xgb_pipeline, X_test, y_test)

    save_model(xgb_pipeline, XGB_PIPELINE_PATH)
