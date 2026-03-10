import joblib
import numpy as np

# Model paths
LR_MODEL_PATH = "models/lr_password_pipeline.pkl"
XGB_MODEL_PATH = "models/xgb_password_pipeline.pkl"

# Label mapping
LABEL_MAP = {0: "Weak", 1: "Medium", 2: "Strong"}


def load_models():
    """Load both trained models"""
    print("[predict] Loading models...")

    lr_model = joblib.load(LR_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)

    return lr_model, xgb_model


def predict_password(model, password: str):
    """Predict password strength"""

    pred = model.predict([password])[0]

    probs = model.predict_proba([password])[0]

    confidence = float(np.max(probs))

    return LABEL_MAP[pred], confidence


if __name__ == "__main__":

    lr_model, xgb_model = load_models()

    print("\nChoose model:")
    print("1 → Logistic Regression")
    print("2 → XGBoost")

    choice = input("Model: ")

    if choice == "1":
        model = lr_model
        print("\nUsing Logistic Regression")
    else:
        model = xgb_model
        print("\nUsing XGBoost")

    while True:

        password = input("\nEnter password (or 'quit'): ")

        if password.lower() == "quit":
            break

        strength, confidence = predict_password(model, password)

        print(f"\nStrength: {strength}")
        print(f"Confidence: {confidence:.3f}")
