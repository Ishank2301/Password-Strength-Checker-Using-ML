import streamlit as st
import joblib
import numpy as np
import math
import time

# Load ML pipeline
MODEL_PATH = "models/lr_password_pipeline.pkl"
model = joblib.load(MODEL_PATH)

LABEL_MAP = {0: "Weak", 1: "Medium", 2: "Strong"}


# Password Entropy
def calculate_entropy(password):

    charset = 0

    if any(c.islower() for c in password):
        charset += 26
    if any(c.isupper() for c in password):
        charset += 26
    if any(c.isdigit() for c in password):
        charset += 10
    if any(c in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~" for c in password):
        charset += 32

    if charset == 0:
        return 0

    entropy = len(password) * math.log2(charset)

    return entropy


# Crack Time Estimator
def estimate_crack_time(entropy):

    guesses_per_second = 1e9  # attacker speed

    guesses = 2**entropy

    seconds = guesses / guesses_per_second

    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.2f} hours"
    elif seconds < 31536000:
        return f"{seconds/86400:.2f} days"
    else:
        return f"{seconds/31536000:.2f} years"


# Page Config
st.set_page_config(
    page_title="AI Password Strength Analyzer", page_icon="🔐", layout="centered"
)

#  Styling
st.markdown(
    """
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
}
.metric-box {
    background-color:#111;
    padding:20px;
    border-radius:10px;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    '<p class="big-title">🔐 AI Password Strength Analyzer</p>', unsafe_allow_html=True
)

st.write("Evaluate password strength using Machine Learning + security metrics.")


password = st.text_input("Enter Password", type="password")


if password:

    pred = model.predict([password])[0]
    probs = model.predict_proba([password])[0]
    confidence = float(np.max(probs))
    strength = LABEL_MAP[pred]

    #  Animated Strength Meter
    progress_bar = st.progress(0)

    for i in range(int(confidence * 100)):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    st.subheader(f"Strength: {strength}")

    if strength == "Weak":
        st.error("⚠️ Weak Password")

    elif strength == "Medium":
        st.warning("⚡ Medium Strength")

    else:
        st.success("✅ Strong Password")

    st.write(f"Model Confidence: **{confidence:.2f}**")

    #  Entropy
    entropy = calculate_entropy(password)

    st.subheader("🔢 Password Entropy")

    st.write(f"Entropy: **{entropy:.2f} bits**")

    entropy_bar = min(entropy / 100, 1.0)

    st.progress(entropy_bar)

    #  Crack Time
    st.subheader("💀 Estimated Crack Time")

    crack_time = estimate_crack_time(entropy)

    st.write(f"Estimated time to brute force: **{crack_time}**")

    # Password Tips
    st.subheader("💡 Suggestions")

    tips = []

    if len(password) < 12:
        tips.append("Use at least 12 characters")

    if not any(c.isupper() for c in password):
        tips.append("Add uppercase letters")

    if not any(c.isdigit() for c in password):
        tips.append("Add numbers")

    if not any(c in "!@#$%^&*" for c in password):
        tips.append("Add special characters")

    if tips:
        for tip in tips:
            st.write(f"- {tip}")
    else:
        st.write("Great password composition!")
