# =============================
# gradient_boosting.py
# Gradient Boosting Injury Prediction (XGBoost + LightGBM)
# Final-project version: detailed explanations and rationale
# =============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from imblearn.combine import SMOTEENN  # Combines SMOTE oversampling + Edited Nearest Neighbors undersampling

def gradient_boosting_pipeline(X, y, model_type="both", threshold=0.05, test_size=0.2, random_state=42):
    """
    Train and evaluate XGBoost and/or LightGBM models for injury prediction.
    Designed for rare-event (imbalanced) injury data, using SMOTE+ENN resampling.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.array
        Target variable (binary: injury)
    model_type : str
        "xgb", "lgb", or "both" – which models to train
    threshold : float
        Probability threshold for binary predictions (default 0.05 to capture rare injuries)
    test_size : float
        Fraction of data reserved for testing
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing trained models, predictions, feature importances, and precision-recall curves
    """

    # -----------------------------
    # 1️⃣ Column name cleaning
    # -----------------------------
    # XGBoost and LightGBM require alphanumeric column names (no spaces/special chars)
    X = X.copy()
    X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True).str.strip('_')

    # -----------------------------
    # 2️⃣ Train-Test Split
    # -----------------------------
    # Stratified split preserves injury ratio in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # -----------------------------
    # 3️⃣ Handle class imbalance (rare injury events)
    # -----------------------------
    # SMOTE: generate synthetic minority samples
    # ENN: remove overlapping/ambiguous majority samples
    smote_enn = SMOTEENN(random_state=random_state)
    X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

    print("Class distribution BEFORE resampling:", np.bincount(y_train))
    print("Class distribution AFTER resampling:", np.bincount(y_train_res))

    results = {}

    # -----------------------------
    # 4️⃣ XGBoost
    # -----------------------------
    if model_type in ("xgb", "both"):
        # XGBoost is robust to missing data and captures nonlinear feature interactions
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,       # Number of boosting rounds
            max_depth=5,            # Limit tree depth to avoid overfitting
            learning_rate=0.05,     # Small step size for gradual learning
            random_state=random_state,
            use_label_encoder=False, # Avoid warning in newer XGBoost versions
            eval_metric='logloss'   # Binary classification metric
        )
        xgb_model.fit(X_train_res, y_train_res)

        # -----------------------------
        # Predictions & Thresholding
        # -----------------------------
        y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
        y_pred_xgb = (y_probs_xgb >= threshold).astype(int)

        # -----------------------------
        # Evaluation Metrics
        # -----------------------------
        print("\n=== XGBoost Evaluation ===")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
        print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
        print("ROC-AUC Score:", roc_auc_score(y_test, y_probs_xgb))

        # -----------------------------
        # Feature Importance
        # -----------------------------
        feat_imp_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop 10 Features (XGBoost):\n", feat_imp_xgb.head(10))

        results['xgb'] = {
            'model': xgb_model,
            'y_probs': y_probs_xgb,
            'y_pred': y_pred_xgb,
            'feature_importances': feat_imp_xgb,
            'precision_recall': precision_recall_curve(y_test, y_probs_xgb)
        }

    # -----------------------------
    # 5️⃣ LightGBM
    # -----------------------------
    if model_type in ("lgb", "both"):
        # LightGBM is optimized for speed and large datasets, handles categorical splits efficiently
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_state
        )
        lgb_model.fit(X_train_res, y_train_res)

        y_probs_lgb = lgb_model.predict_proba(X_test)[:, 1]
        y_pred_lgb = (y_probs_lgb >= threshold).astype(int)

        print("\n=== LightGBM Evaluation ===")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lgb))
        print("\nClassification Report:\n", classification_report(y_test, y_pred_lgb))
        print("ROC-AUC Score:", roc_auc_score(y_test, y_probs_lgb))

        feat_imp_lgb = pd.Series(lgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop 10 Features (LightGBM):\n", feat_imp_lgb.head(10))

        results['lgb'] = {
            'model': lgb_model,
            'y_probs': y_probs_lgb,
            'y_pred': y_pred_lgb,
            'feature_importances': feat_imp_lgb,
            'precision_recall': precision_recall_curve(y_test, y_probs_lgb)
        }

    return results


# ==============================
# Example usage (if run as script)
# ==============================
if __name__ == "__main__":
    # Load preprocessed injury dataset
    df = pd.read_csv("data/merged_tracking_injury_features.csv")
    X = df.drop(columns=['injury', 'injury_type'])
    y = df['injury']

    # Train both models and evaluate
    gradient_boosting_pipeline(X, y, model_type="both")