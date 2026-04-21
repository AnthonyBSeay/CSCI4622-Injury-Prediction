# =============================
# util.py
# Model Comparison Visualization Tools
# =============================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

sns.set_style("whitegrid")


# =============================
# Helper: Compute Metrics
# =============================
def compute_metrics(y_true, y_pred, y_proba):
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "PR_AUC": pr_auc
    }

# =============================
# 1. ROC Curve Comparison
# =============================
def plot_roc_curves(model_outputs):
    """
    model_outputs: list of dicts
    Each dict must contain:
        {
            "name": str,
            "y_true": array,
            "y_proba": array
        }
    """
    plt.figure(figsize=(8, 6))

    for model in model_outputs:
        fpr, tpr, _ = roc_curve(model["y_true"], model["y_proba"])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{model['name']} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================
# 2. Precision-Recall Curve
# =============================
def plot_pr_curves(model_outputs):
    plt.figure(figsize=(8, 6))

    y_true = model_outputs[0]["y_true"]
    baseline = sum(y_true) / len(y_true)

    for model in model_outputs:
        precision, recall, _ = precision_recall_curve(
            model["y_true"], model["y_proba"]
        )
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision,
                 label=f"{model['name']} (PR AUC = {pr_auc:.3f})")
        
    plt.hlines(baseline, 0, 1, linestyles="dashed", label=f"Baseline ({baseline:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================
# 3. Metrics Bar Chart
# =============================
def plot_metrics_bar(model_outputs):
    model_names = []
    metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "PR_AUC": []
    }

    for model in model_outputs:
        m = compute_metrics(
            model["y_true"],
            model["y_pred"],
            model["y_proba"]
        )

        model_names.append(model["name"])
        for key in metrics:
            metrics[key].append(m[key])

    x = np.arange(len(model_names))
    width = 0.15  # slightly smaller for more bars

    plt.figure(figsize=(12, 6))

    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.bar(x + i * width, values, width, label=metric_name)

    plt.xticks(x + width * 2, model_names)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison (Including PR AUC)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================
# 4. Confusion Matrix Grid
# =============================
def plot_confusion_matrices(model_outputs):
    n = len(model_outputs)
    cols = 2
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(10, 4 * rows))

    for i, model in enumerate(model_outputs):
        cm = confusion_matrix(model["y_true"], model["y_pred"])

        plt.subplot(rows, cols, i + 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title(model["name"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()


# =============================
# MASTER FUNCTION
# =============================
def compare_all_models(model_outputs):
    """
    Runs all comparison plots at once
    """
    plot_roc_curves(model_outputs)
    plot_pr_curves(model_outputs)
    plot_metrics_bar(model_outputs)
    plot_confusion_matrices(model_outputs)