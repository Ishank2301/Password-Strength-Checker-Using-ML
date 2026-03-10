from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Password Strength API")

# Load trained pipeline:
model = joblib.load("models/lr_password_model.pkl")

# Mapping the Labels:]
LABEL_MAP = {
    0: "Weak",
    1: "Medium",
    2: "Strong",
}


# Request schema:
class PasswordRequest(BaseModel):
    password: str


@app.get("/")
def home():
    return {"message": "Password Strength API is running"}


@app.post("/predict")
def predict_strength(request: PasswordRequest):
    password = request.password
    prediction = model.predict([password])[0]
    probabilities = model.predict_proba([password])[0]
    confidence = float(np.max(probabilities))

    return {
        "password": password,
        "strength": LABEL_MAP[prediction],
        "confidence": round(confidence, 3),
    }
