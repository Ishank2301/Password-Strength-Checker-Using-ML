import joblib

LABEL_MAP = {0: "Weak", 1: "Medium", 2: "Strong"}


def save_model(model, path):

    joblib.dump(model, path)
    print(f"[helper] Model saved to {path}")


def load_model(path):

    model = joblib.load(path)
    print(f"[helper] Model loaded from {path}")

    return model


def decode_label(label):

    return LABEL_MAP.get(label, "Unknown")
