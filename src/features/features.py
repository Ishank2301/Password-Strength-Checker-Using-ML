import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# Creating a vectorizer function:
def create_vectorizer():

    return TfidfVectorizer(
        analyzer="char", ngram_range=(2, 3), sublinear_tf=True, max_features=5000
    )


def train_vectorizer():

    print("[features] Loading processed dataset...")

    df = pd.read_csv("data/processed/password_features.csv")

    passwords = df["password"]

    vectorizer = create_vectorizer()

    X_vec = vectorizer.fit_transform(passwords)

    print(f"[features] Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"[features] Vectorized shape: {X_vec.shape}")

    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"[features] Vectorizer saved: {VECTORIZER_PATH}")


if __name__ == "__main__":
    train_vectorizer()
