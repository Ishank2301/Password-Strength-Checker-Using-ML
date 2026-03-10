import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

LR_MODEL_PATH = "models/logistic_model.pkl"
XGB_MODEL_PATH = "models/XG_boost_model.pkl"


def load_data():
    print("[train] Loading dataset...")

    df = pd.read_csv("data/processed/password_features.csv")

    X = df["password"]
    y = df["strength"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def load_vectorizer():

    print("[train] Loading TF-IDF vectorizer...")

    vectorizer = joblib.load(VECTORIZER_PATH)

    return vectorizer


def train_logistic_regression(X_train_vec, y_train):

    print("[train] Training Logistic Regression...")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vec, y_train)

    return model


def train_xgboost(X_train_vec, y_train):
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1
    )
    model.fit(X_train_vec, y_train)
    return model


def evaluate_model(model, X_test_vec, y_test):

    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)

    print(f"\nAccuracy: {acc:.4f}\n")

    print(classification_report(y_test, preds))


def save_model(model, path):

    joblib.dump(model, path)

    print(f"[train] Model saved : {path}")


if __name__ == "__main__":

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Load vectorizer
    vectorizer = load_vectorizer()

    # Convert passwords → vectors
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"[train] Train matrix shape: {X_train_vec.shape}")

    # Train models
    lr_model = train_logistic_regression(X_train_vec, y_train)
    xgb_model = train_xgboost(X_train_vec, y_train)

    # Evaluate
    print("\n--- Logistic Regression Results ---")
    evaluate_model(lr_model, X_test_vec, y_test)

    print("\n--- Gradient Boosting Results ---")
    evaluate_model(xgb_model, X_test_vec, y_test)

    # Save models
    save_model(lr_model, LR_MODEL_PATH)
    save_model(xgb_model, XGB_MODEL_PATH)
