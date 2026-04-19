# =============================
# RNN.py (Improved Threshold Tuning Version)
# Injury Prediction using LSTM (PyTorch)
# =============================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt


# -----------------------------
# 1️⃣ Dataset
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
# 2️⃣ LSTM Model
# -----------------------------
class InjuryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# -----------------------------
# 3️⃣ Sequence Builder
# -----------------------------
def create_sequences(df, feature_cols, target_col, sequence_length=5, horizon=5):
    sequences = []
    labels = []

    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    for _, group in df.groupby("PLAYER_ID"):
        group = group.reset_index(drop=True)

        for i in range(len(group) - sequence_length - horizon):

            seq = group[feature_cols].iloc[i:i+sequence_length].values

            future_injuries = group[target_col].iloc[
                i+sequence_length : i+sequence_length+horizon
            ].values

            label = int(np.any(future_injuries == 1))

            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)


# -----------------------------
# 4️⃣ Train/Test Split
# -----------------------------
def train_test_split_players(df, test_size=0.2, seed=42):
    np.random.seed(seed)

    players = df["PLAYER_ID"].unique()
    np.random.shuffle(players)

    split = int(len(players) * (1 - test_size))
    train_players = players[:split]
    test_players = players[split:]

    train_df = df[df["PLAYER_ID"].isin(train_players)].copy()
    test_df = df[df["PLAYER_ID"].isin(test_players)].copy()

    return train_df, test_df


# -----------------------------
# 5️⃣ Training Pipeline
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

    # Split
    train_df, test_df = train_test_split_players(df, seed=random_state)

    # Scaling
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Sequences
    X_train, y_train = create_sequences(train_df, feature_cols, target_col, sequence_length)
    X_test, y_test = create_sequences(test_df, feature_cols, target_col, sequence_length)

    print(f"Train sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")

    # Dataloaders
    train_loader = DataLoader(InjuryDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(InjuryDataset(X_test, y_test), batch_size=batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = InjuryLSTM(input_size=len(feature_cols)).to(device)

    # Class imbalance
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([neg / (pos + 1e-6)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
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

    # -----------------------------
    # Threshold tuning (NEW)
    # -----------------------------
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    thresholds = np.append(thresholds, 1.0)

    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    threshold_results = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    # F1 optimal
    f1_idx = threshold_results["f1"].idxmax()
    f1_threshold = threshold_results.loc[f1_idx, "threshold"]

    # High recall (>= 0.80)
    recall_target = 0.80
    valid_recall = threshold_results[threshold_results["recall"] >= recall_target]

    if len(valid_recall) > 0:
        idx = valid_recall["precision"].idxmax()
        recall_threshold = threshold_results.loc[idx, "threshold"]
    else:
        recall_threshold = f1_threshold

    # Higher precision (>= 0.25)
    precision_target = 0.25
    valid_precision = threshold_results[threshold_results["precision"] >= precision_target]

    if len(valid_precision) > 0:
        idx = valid_precision["recall"].idxmax()
        precision_threshold = threshold_results.loc[idx, "threshold"]
    else:
        precision_threshold = f1_threshold

    # -----------------------------
    # Evaluation function
    # -----------------------------
    def evaluate_threshold(name, threshold):
        preds = (probs >= threshold).astype(int)

        print(f"\n=== {name} (threshold={threshold:.4f}) ===")
        print("Confusion Matrix:\n", confusion_matrix(y_true, preds))
        print("\nClassification Report:\n",
              classification_report(y_true, preds, zero_division=0))

    evaluate_threshold("F1 Optimal", f1_threshold)
    evaluate_threshold("High Recall (>=0.80)", recall_threshold)
    evaluate_threshold("Higher Precision", precision_threshold)

    # -----------------------------
    # PR-AUC
    # -----------------------------
    pr_auc = auc(recall, precision)
    print("\nPR-AUC:", pr_auc)

    # -----------------------------
    # Plot threshold curves
    # -----------------------------
    plt.figure()
    plt.plot(threshold_results["threshold"], threshold_results["precision"], label="Precision")
    plt.plot(threshold_results["threshold"], threshold_results["recall"], label="Recall")
    plt.plot(threshold_results["threshold"], threshold_results["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning Curve")
    plt.legend()
    plt.show()

    return {
        "model": model,
        "scaler": scaler,
        "y_true": y_true,
        "y_probs": probs,
        "f1_threshold": f1_threshold,
        "recall_threshold": recall_threshold,
        "precision_threshold": precision_threshold,
        "pr_auc": pr_auc
    }