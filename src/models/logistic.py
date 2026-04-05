# logistic.py
# =============================
# Logistic Regression Injury Model (SMOTE + Undersampling)
# =============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve
)
from imblearn.combine import SMOTEENN


def train_logistic_model_from_csv(
    csv_path: str,
    target_col: str = "injury",
    feature_cols: list = None,
    test_size: float = 0.2,
    threshold: float = 0.05,
    random_state: int = 42,
    verbose: bool = True
):
    """
    Train a logistic regression model on CSV data using SMOTE + undersampling.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing features and target
    target_col : str
        Column name of the target variable
    test_size : float
        Fraction of data to hold out for testing
    threshold : float
        Probability threshold for predicting positive class
    random_state : int
        Random seed for reproducibility
    verbose : bool
        If True, prints metrics and feature importance

    Returns
    -------
    dict
        Dictionary containing trained model, scaler, test sets, predictions, and metrics
    """

    # -----------------------------
    # 0️⃣ Load data
    # -----------------------------
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop non-feature columns
    drop_cols = [
        "SEASON_ID", "PLAYER_ID", "GAME_ID", "GAME_DATE",
        "MATCHUP", "injury_type", "gameId", "personId", "prev_game_date"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # -----------------------------
    # 1️⃣ Split features & target
    # -----------------------------
    # Split features & target
    y = df[target_col].astype(int)

    if feature_cols is not None:
        X = df[feature_cols]
    else:
        drop_cols = [
            "SEASON_ID", "PLAYER_ID", "GAME_ID", "GAME_DATE",
            "MATCHUP", "injury_type", "gameId", "personId", "prev_game_date"
        ]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns and c != target_col])

    # -----------------------------
    # 2️⃣ Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # -----------------------------
    # 3️⃣ SMOTE + Undersampling
    # -----------------------------
    smote_enn = SMOTEENN(random_state=random_state)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    if verbose:
        print("Before Resampling:", np.bincount(y_train))
        print("After Resampling:", np.bincount(y_train_resampled))

    # -----------------------------
    # 4️⃣ Feature Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # 5️⃣ Train Logistic Regression
    # -----------------------------
    logreg = LogisticRegression(
        class_weight=None,  # OFF since we resampled
        max_iter=1000,
        random_state=random_state
    )
    logreg.fit(X_train_scaled, y_train_resampled)

    # -----------------------------
    # 6️⃣ Predictions
    # -----------------------------
    y_probs = logreg.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # -----------------------------
    # 7️⃣ Evaluation
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_probs)
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    # -----------------------------
    # 8️⃣ Feature Importance
    # -----------------------------
    feature_importance = pd.Series(
        np.abs(logreg.coef_[0]), index=X.columns
    ).sort_values(ascending=False)

    if verbose:
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nROC-AUC Score:", roc_auc)
        print("\nTop 10 Features Influencing Injury Risk:\n", feature_importance.head(10))

    return {
        "model": logreg,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "confusion_matrix": cm,
        "classification_report": cr,
        "roc_auc": roc_auc,
        "precision_recall_curve": (precision, recall, thresholds),
        "feature_importance": feature_importance
    }


# Example usage
if __name__ == "__main__":
    results = train_logistic_model_from_csv("data/model_ready_data.csv", verbose=True)