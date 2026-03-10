"""
Trains Logistic Regression and Gradient Boosting classifiers
on the TF-IDF vectorized password dataset.
"""

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

LR_MODEL_PATH = "models/logistic_model.pkl"
GB_MODEL_PATH = "models/gradient_boost_model.pkl"


def train_logistic_regression(X_train, y_train):
    print("[train] Training Logistic Regression...")

    model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)
    model.fit(X_train, y_train)
    print("[train] Logistic Regression training complete.")
    return model


def train_gradient_boosting(X_train, y_train):
    print("[train] Training Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    print("[train] Gradient Boosting training complete.")
    return model


def save_model(model, path):
    joblib.dump(model, path)
    print(f"[train] Model saved to {path}")


def load_model(path):
    model = joblib.load(path)
    print(f"[train] Model loaded from {path}")
    return model


if __name__ == "__main__":

    print("[train] Loading dataset...")

    df = pd.read_csv("data/processed/password_features.csv")

    X = df.drop("strength", axis=1)
    y = df["strength"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_model = train_logistic_regression(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    save_model(lr_model, LR_MODEL_PATH)
    save_model(gb_model, GB_MODEL_PATH)
