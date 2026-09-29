import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import SEED


def load_academic_performance_data(n_samples=200):
    """
    Loads UCI Student Performance dataset.

    Binary classification (kept identical to the original VQC task
    so the two projects are directly comparable):
        Pass (+1): G3 >= 10
        Fail (-1): G3 < 10
    """

    np.random.seed(SEED)

    print("Loading UCI Student Performance dataset...")

    df = pd.read_csv(
        "data/student-mat.csv",
        sep=";"
    )

    feature_cols = [
        "studytime",
        "failures",
        "absences",
        "G1"
    ]

    X = df[feature_cols].values

    y = np.where(
        df["G3"].values >= 10,
        1.0,
        -1.0
    )

    if n_samples < len(X):

        indices = np.random.choice(
            len(X),
            size=n_samples,
            replace=False
        )

        X = X[indices]
        y = y[indices]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_academic_performance_data()

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
