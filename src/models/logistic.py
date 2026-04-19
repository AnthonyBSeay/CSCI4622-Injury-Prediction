# =============================
# logistic.py
# Logistic Regression Injury Model (SMOTE + Undersampling)
# Final-project version: detailed explanations and rationale
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
from imblearn.combine import SMOTEENN  # Handles class imbalance using oversampling + cleaning


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
    Train a logistic regression model on CSV data using SMOTE + ENN undersampling.
    Designed for rare-event injury prediction with class imbalance.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing features and target.
    target_col : str
        Column name of the target variable (binary: injury/no injury).
    feature_cols : list
        Optional subset of columns to use as features. If None, auto-select all except dropped columns.
    test_size : float
        Fraction of data to hold out for testing (default 0.2).
    threshold : float
        Probability threshold for predicting positive class (injury).
        Default 0.05 is low to catch rare injuries.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        If True, prints metrics and feature importance.

    Returns
    -------
    dict
        Dictionary containing trained model, scaler, test sets, predictions, and evaluation metrics.
    """

    # -----------------------------
    # 0️⃣ Load data
    # -----------------------------
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop irrelevant/non-feature columns that do not contribute to prediction
    drop_cols = [
        "SEASON_ID", "PLAYER_ID", "GAME_ID", "GAME_DATE",
        "MATCHUP", "injury_type", "gameId", "personId", "prev_game_date"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove rows with missing target
    df = df.dropna(subset=[target_col])

    # -----------------------------
    # 1️⃣ Split features & target
    # -----------------------------
    # Target variable (binary)
    y = df[target_col].astype(int)

    # Features
    if feature_cols is not None:
        X = df[feature_cols]
    else:
        # Auto-select all columns except dropped ones
        drop_cols = [
            "SEASON_ID", "PLAYER_ID", "GAME_ID", "GAME_DATE",
            "MATCHUP", "injury_type", "gameId", "personId", "prev_game_date"
        ]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns and c != target_col])

    # -----------------------------
    # 2️⃣ Train-Test Split
    # -----------------------------
    # Stratify ensures class balance is preserved in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # -----------------------------
    # 3️⃣ Handle class imbalance: SMOTE + ENN
    # -----------------------------
    # Rare injuries may bias the model towards predicting 'no injury'.
    # SMOTE: oversample minority class by synthetic generation.
    # ENN: removes ambiguous/overlapping samples to reduce noise.
    smote_enn = SMOTEENN(random_state=random_state)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    if verbose:
        print("Before Resampling:", np.bincount(y_train))
        print("After Resampling:", np.bincount(y_train_resampled))
        # Useful to show how class balance improved

    # -----------------------------
    # 4️⃣ Feature Scaling
    # -----------------------------
    # Standardization improves convergence and avoids coefficient scaling issues
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # 5️⃣ Train Logistic Regression
    # -----------------------------
    # No class_weight since we've resampled
    logreg = LogisticRegression(
        class_weight=None,
        max_iter=1000,  # Ensure convergence
        random_state=random_state
    )
    logreg.fit(X_train_scaled, y_train_resampled)

    # -----------------------------
    # 6️⃣ Predictions
    # -----------------------------
    # Predict probabilities for positive class
    y_probs = logreg.predict_proba(X_test_scaled)[:, 1]
    # Apply custom threshold (low for rare-event detection)
    y_pred = (y_probs >= threshold).astype(int)

    # -----------------------------
    # 7️⃣ Evaluation Metrics
    # -----------------------------
    # Confusion matrix, classification report, ROC-AUC, PR curve
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_probs)
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    # PR curve is more informative than ROC for imbalanced datasets

    # -----------------------------
    # 8️⃣ Feature Importance
    # -----------------------------
    # Absolute values of coefficients reflect impact on injury prediction
    feature_importance = pd.Series(
        np.abs(logreg.coef_[0]), index=X.columns
    ).sort_values(ascending=False)

    if verbose:
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nROC-AUC Score:", roc_auc)
        print("\nTop 10 Features Influencing Injury Risk:\n", feature_importance.head(10))

    # -----------------------------
    # 🔟 Additional Evaluation Metrics (IMPORTANT FOR REPORT)
    # -----------------------------

    tn, fp, fn, tp = cm.ravel()

    # Core derived metrics
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    balanced_accuracy = ( (tp / (tp + fn)) + specificity ) / 2

    # PR-AUC (very important for imbalanced data)
    pr_auc = auc(recall, precision)

    if verbose:
        print("\n==============================")
        print("ADDITIONAL EVALUATION METRICS")
        print("==============================")

        print(f"Specificity (TNR): {specificity:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"PR-AUC Score: {pr_auc:.4f}")

    # -----------------------------
    # Threshold Sensitivity Analysis
    # -----------------------------
    thresholds_to_test = np.arange(0.01, 0.51, 0.01)

    best_f1 = 0
    best_threshold = threshold

    for t in thresholds_to_test:
        preds = (y_probs >= t).astype(int)
        report = classification_report(y_test, preds, output_dict=True)
        f1 = report["1"]["f1-score"]

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    if verbose:
        print("\n==============================")
        print("THRESHOLD OPTIMIZATION")
        print("==============================")
        print(f"Best Threshold (F1-based): {best_threshold:.3f}")
        print(f"Best F1 Score: {best_f1:.4f}")


    # -----------------------------
    # Return model and results for analysis or reporting
    # -----------------------------
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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_logistic_results(results: dict, model_name: str = "Logistic Regression", top_n_features: int = 10):

    y_test = results["y_test"]
    y_probs = results["y_probs"]
    cm = results["confusion_matrix"]
    precision, recall, pr_thresholds = results["precision_recall_curve"]
    roc_auc = results["roc_auc"]
    feature_importance = results["feature_importance"]

    # =========================================================
    # 1️⃣ CONFUSION MATRIX
    # =========================================================
    plt.figure()
    plt.imshow(cm)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()

    # =========================================================
    # 2️⃣ ROC CURVE
    # =========================================================
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # =========================================================
    # 3️⃣ PRECISION-RECALL CURVE
    # =========================================================
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    # =========================================================
    # 4️⃣ THRESHOLD TRADEOFF PLOT (NEW)
    # =========================================================
    thresholds = np.linspace(0.01, 0.99, 100)

    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)

        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))

        precision_t = tp / (tp + fp + 1e-8)
        recall_t = tp / (tp + fn + 1e-8)
        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-8)

        precisions.append(precision_t)
        recalls.append(recall_t)
        f1s.append(f1_t)

    plt.figure()
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1 Score")

    plt.title(f"{model_name} - Threshold Tradeoff")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    # =========================================================
    # 5️⃣ FEATURE IMPORTANCE
    # =========================================================
    top_features = feature_importance.head(top_n_features)

    plt.figure()
    top_features.sort_values().plot(kind="barh")
    plt.title(f"{model_name} - Top {top_n_features} Features")
    plt.xlabel("Importance (|Coefficient|)")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # This allows direct testing when running the file
    results = train_logistic_model_from_csv("data/model_ready_data.csv", verbose=True)