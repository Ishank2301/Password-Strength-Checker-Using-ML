# Performing Feature Engineering to Extract to convert data into vector
from sklearn.feature_extraction.text import Tfidfvectorizer

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def create_vectorizer():
    vectorizer = Tfidfvectorizer(
        analyzer="char", ngram_range=(1, 3)
    )  #  Password strength detection works best with character n-grams.

    return vectorizer
