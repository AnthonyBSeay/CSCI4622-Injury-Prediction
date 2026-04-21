# =============================
# RNN.py 
# =============================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt


# -----------------------------
# Dataset
# -----------------------------
class InjuryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# LSTM Model
# -----------------------------
class InjuryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# -----------------------------
# Sequence Builder
# -----------------------------
def create_sequences(df, feature_cols, target_col, sequence_length=5, horizon=5):
    sequences, labels = [], []

    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    for _, group in df.groupby("PLAYER_ID"):
        group = group.reset_index(drop=True)

        for i in range(len(group) - sequence_length - horizon):
            seq = group[feature_cols].iloc[i:i+sequence_length].values
            future = group[target_col].iloc[i+sequence_length:i+sequence_length+horizon].values
            label = int(np.any(future == 1))

            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)


# -----------------------------
# Train/Test Split
# -----------------------------
def train_test_split_players(df, test_size=0.2, seed=42):
    np.random.seed(seed)

    players = df["PLAYER_ID"].unique()
    np.random.shuffle(players)

    split = int(len(players) * (1 - test_size))
    train_players = players[:split]
    test_players = players[split:]

    return (
        df[df["PLAYER_ID"].isin(train_players)].copy(),
        df[df["PLAYER_ID"].isin(test_players)].copy()
    )


# -----------------------------
# 🔥 Threshold Optimization
# -----------------------------
def find_optimal_threshold(y_true, y_probs, min_recall=0.6):
    thresholds = np.linspace(0.005, .995, 200)

    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)

        if recall >= min_recall:
            if precision > best_precision:
                best_precision = precision
                best_threshold = t
                best_recall = recall

    return best_threshold, best_precision, best_recall


# -----------------------------
# Training Pipeline
# -----------------------------
def train_rnn_from_csv(
    csv_path,
    feature_cols,
    target_col="injury",
    sequence_length=5,
    batch_size=64,
    epochs=10,
    lr=0.001,
    random_state=42
):

    print("Loading data...")
    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split_players(df, seed=random_state)

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    X_train, y_train = create_sequences(train_df, feature_cols, target_col, sequence_length)
    X_test, y_test = create_sequences(test_df, feature_cols, target_col, sequence_length)

    print(f"Train sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")

    train_loader = DataLoader(InjuryDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(InjuryDataset(X_test, y_test), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = InjuryLSTM(input_size=len(feature_cols)).to(device)

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([neg / (pos + 1e-6)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training RNN...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()
    logits_list, y_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze().cpu().numpy()

            logits_list.extend(logits)
            y_true.extend(y_batch.numpy())

    logits = np.array(logits_list)
    y_true = np.array(y_true)

    probs = 1 / (1 + np.exp(-logits))

    # 🔥 OPTIMAL THRESHOLD
    best_threshold, best_precision, best_recall = find_optimal_threshold(y_true, probs)

    preds = (probs >= best_threshold).astype(int)

    print(f"\n=== FINAL MODEL (Optimized Threshold) ===")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Precision: {best_precision:.3f}")
    print(f"Recall: {best_recall:.3f}")

    cm = confusion_matrix(y_true, preds)
    print("Confusion Matrix:\n", cm)

    print("\nClassification Report:\n",
          classification_report(y_true, preds, zero_division=0))

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall_curve, precision_curve)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    return {
        "model": model,
        "threshold": best_threshold,
        "y_true": y_true,
        "y_probs": probs,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

def plot_rnn_results(results, model_name="RNN"):
    """
    Plots:
    - ROC Curve
    - Precision-Recall Curve
    - Threshold Tradeoff Curve (Precision, Recall vs Threshold)
    """

    y_true = results["y_true"]
    y_probs = results["y_probs"]

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # -----------------------------
    # Precision-Recall Curve
    # -----------------------------
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    # -----------------------------
    # Threshold Tradeoff Curve
    # -----------------------------
    thresholds = np.linspace(0.0, 1.0, 100)

    precision_scores = []
    recall_scores = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)

        precision_scores.append(prec)
        recall_scores.append(rec)

    plt.figure()
    plt.plot(thresholds, precision_scores, label="Precision")
    plt.plot(thresholds, recall_scores, label="Recall")
    plt.title(f"{model_name} - Threshold Tradeoff")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

