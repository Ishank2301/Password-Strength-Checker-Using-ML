import pandas as pd
from sklearn.model_selection import train_test_split

# Mapping Password Strength in Levels:
LABEL_MAP = {0: "Weak", 1: "Medium", 2: "Strong"}


def load_data(filepath: str) -> pd.DataFrame:
    # Load the CSV and return a cleaned dataframe
    df = pd.read_csv(
        "D:\Ai ml\Projects_ALL\Password-Strength-Checker-Using-ML\data\raw\Password_Strength.csv",
        on_bad_lines="skip",
    )

    # Perform EDA on the Data:
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop rows with missing values:
    before = len(df)
    df = df.dropna(subset=["password", "strength"])
    after = len(df)
    print(f"[preprocess] Dropped {before - after} rows with missing values.")
    print(f"[preprocess] Dataset size: {after} rows.")

    # Ensure correct types
    df["password"] = df["password"].astype(str)
    df["strength"] = df["strength"].astype(int)

    return df


def get_class_distributions(df: pd.DataFrame) -> pd.Series:
    # Return value counts of strength label:
    return df["strength"].map(LABEL_MAP).value_counts()


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    # Split dataset into train and test set:
    X = df["password"]
    y = df["strength"]

    # Divide data into training and test sets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[preprocess] Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test
