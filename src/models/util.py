import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_gbm_or_any_model(y_test, y_probs, model_name="Model"):
    """
    Unified evaluation plots for XGBoost / LightGBM / Logistic Regression / RNN outputs.

    Parameters
    ----------
    y_test : array-like
        True binary labels
    y_probs : array-like
        Predicted probabilities for positive class
    model_name : str
        Name shown in plot titles
    """

    # =========================================================
    # 1️⃣ ROC CURVE
    # =========================================================
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # =========================================================
    # 2️⃣ PRECISION-RECALL CURVE
    # =========================================================
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    # =========================================================
    # 3️⃣ THRESHOLD TRADEOFF PLOT
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