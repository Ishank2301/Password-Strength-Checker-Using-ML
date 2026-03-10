import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {0: "Weak", 1: "Medium", 2: "Strong"}


def load_data(filepath: str) -> pd.DataFrame:

    df = pd.read_csv(
        filepath,
        on_bad_lines="skip",
    )

    df.columns = [c.strip().lower() for c in df.columns]

    before = len(df)
    df = df.dropna(subset=["password", "strength"])
    after = len(df)

    print(f"[preprocess] Dropped {before - after} rows with missing values.")
    print(f"[preprocess] Dataset size: {after} rows.")

    df["password"] = df["password"].astype(str)
    df["strength"] = df["strength"].astype(int)

    return df


def get_class_distributions(df: pd.DataFrame):

    return df["strength"].map(LABEL_MAP).value_counts()


def save_processed_data(df: pd.DataFrame):

    output_path = "data/processed/password_features.csv"
    df.to_csv(output_path, index=False)

    print(f"[preprocess] Processed data saved to {output_path}")


if __name__ == "__main__":

    df = load_data("data/raw/Password_Strength.csv")

    print("\n[preprocess] Class distribution:")
    print(get_class_distributions(df))

    save_processed_data(df)
