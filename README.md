# 🔐 Password Strength Classifier using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-670K%20Passwords-purple?style=flat-square)

> A machine learning pipeline that classifies password strength into **Weak**, **Medium**, or **Strong** using character-level TF-IDF features with Logistic Regression and Gradient Boosting classifiers.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Design](#-system-design)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Usage](#-usage)
- [Dependencies](#-dependencies)
- [Acknowledgements](#-acknowledgements)

---

## 🧠 Overview

Password security is a critical component of digital safety. This project builds a supervised machine learning model trained on 670,000 real-world passwords to predict password strength. It leverages **character-level TF-IDF vectorization** to extract meaningful patterns from raw password strings and feeds them into classification models.

**Key Capabilities:**
- Classifies passwords as `Weak (0)`, `Medium (1)`, or `Strong (2)`
- Trained on a diverse, real-world dataset of 670K passwords
- Supports interactive strength prediction via CLI
- Evaluates models using Accuracy, Precision, Recall, and F1-Score

---

## 🏗 System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│              Raw Password String (e.g. "P@ssw0rd!")              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                           │
│  • Drop null/missing rows                                        │
│  • Label encoding: Weak=0, Medium=1, Strong=2                    │
│  • Train/Test Split (80/20)                                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                       │
│  • Character-level TF-IDF Vectorizer                             │
│  • Analyzer: 'char'  |  N-gram Range: (1, 3)                    │
│  • Captures: digits, symbols, uppercase, lowercase patterns      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                 │
│                                                                  │
│   ┌─────────────────────┐    ┌──────────────────────────────┐   │
│   │  Logistic Regression│    │  Gradient Boosting Classifier│   │
│   │  (Baseline Model)   │    │  (Ensemble Model)            │   │
│   └─────────┬───────────┘    └──────────────┬───────────────┘   │
│             └──────────────┬────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     EVALUATION LAYER                             │
│  Accuracy | Precision | Recall | F1-Score | Confusion Matrix     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│         Predicted Class: Weak / Medium / Strong                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Source:** [Kaggle - Password Strength Dataset](https://www.kaggle.com/)

| Property       | Details                          |
|----------------|----------------------------------|
| Total Records  | ~670,000 passwords               |
| Features       | `password`, `strength`           |
| Target Classes | 0 = Weak, 1 = Medium, 2 = Strong |
| Missing Values | Dropped during preprocessing     |

**Strength Label Criteria:**

| Label | Class   | Example Characteristics                          |
|-------|---------|--------------------------------------------------|
| `0`   | Weak    | Short, no symbols, dictionary words              |
| `1`   | Medium  | Mix of letters/numbers, moderate length          |
| `2`   | Strong  | Long, symbols, digits, mixed case, high entropy  |

---

## ⚙️ Feature Engineering

Raw passwords are converted to numerical vectors using **character-level TF-IDF**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    analyzer='char',        # Character-level tokenization
    ngram_range=(1, 3),     # Unigrams, bigrams, trigrams
    sublinear_tf=True       # Apply log normalization
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
```

**Why character-level TF-IDF?**
- Captures structural patterns like `@`, `!`, `123`, `aaa`
- No need for manual feature crafting (length, digit count, etc.)
- Generalizes well to unseen password structures

---

## 🤖 Model Architecture

### 1. Logistic Regression *(Baseline)*
- Fast, interpretable
- Suitable for high-dimensional sparse TF-IDF vectors
- Multi-class via `softmax` / one-vs-rest

### 2. Gradient Boosting Classifier *(Ensemble)*
- Builds sequential decision trees to minimize classification error
- Higher accuracy at the cost of training time
- Handles non-linear patterns in character sequences

---

## 📁 Project Structure
Create the Project Folder Structure

From your terminal inside Projects_ALL:
 
---mkdir password-strength-ml 
---cd password-strength-ml

Now create folders:
 
---mkdir data 
---mkdir data/raw 
---mkdir data/processed
 
---mkdir notebooks
 
---mkdir src 
---mkdir src/data 
---mkdir src/features 
---mkdir src/models 
---mkdir src/utils
 
---mkdir models 
---mkdir config 
---mkdir tests
 
---mkdir api 
---mkdir ui

Now create files:
 
---touch README.md 
---touch requirements.txt 
---touch main.py 
---touch .gitignore
 
---touch config/config.yaml
 
---touch src/data/preprocess.py 
---touch src/features/features.py 
---touch src/models/train.py 
---touch src/models/evaluate.py 
---touch src/models/predict.py 
---touch src/utils/helper.py
 
---touch tests/test_pipeline.py
 
---touch api/app.py 
---touch ui/app.py

Final structure becomes:

password-strength-ml
│
├── api
│   └── app.py
│
├── ui
│   └── app.py
│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│
├── src
│   ├── data
│   ├── features
│   ├── models
│   └── utils
│
├── models
├── config
├── test ---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/password-strength-ml.git
cd password-strength-ml

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Train models
python src/train.py

# Evaluate performance
python src/evaluate.py

# Predict password strength interactively
python src/predict.py
```

---

## 📈 Results

| Model                    | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | ~X%      | ~X%       | ~X%    | ~X%      |
| Gradient Boosting        | ~X%      | ~X%       | ~X%    | ~X%      |

> ⚠️ Replace the placeholder values above with your actual results after training.

---

## 🖥️ Usage

Run the interactive password checker:

```bash
python src/predict.py
```

**Example:**

```
Enter a password: MyDog$Name2024!
Predicted Strength: Strong 💪
```

You can also use it programmatically:

```python
from src.predict import predict_strength

result = predict_strength("hello123")
print(result)  # Output: "Medium"
```

---

## 📦 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
seaborn>=0.11.0
matplotlib>=3.4.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🙏 Acknowledgements

- [Kaggle Password Strength Dataset](https://www.kaggle.com/) — for the labeled password corpus
- [scikit-learn](https://scikit-learn.org/) — ML models and TF-IDF vectorizer
- [Seaborn](https://seaborn.pydata.org/) — data visualization
- The open-source community for their contributions to Python data science tooling

---

## 📄 License

This project is licensed under the [General License](LICENSE).

---


<p align="center">
  Made with ❤️ | Star ⭐ this repo if you found it useful!
</p>