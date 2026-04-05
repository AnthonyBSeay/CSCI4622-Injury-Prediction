# =============================
# RNN.py
# Injury Prediction using LSTM (PyTorch)
# =============================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve

# -----------------------------
# 1️⃣ Sequence Dataset
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)


# -----------------------------
# 3️⃣ Create Sequences
# -----------------------------
def create_sequences(df, feature_cols, target_col, sequence_length=5):
    sequences = []
    labels = []

    # ⚠️ MUST have player_id + date
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    for player_id, group in df.groupby("PLAYER_ID"):
        group = group.reset_index(drop=True)

        for i in range(len(group) - sequence_length):
            seq = group[feature_cols].iloc[i:i+sequence_length].values
            label = group[target_col].iloc[i+sequence_length]

            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)


# -----------------------------
# 4️⃣ Training Pipeline
# -----------------------------
def train_rnn_from_csv(
    csv_path,
    feature_cols,
    target_col="injury",
    sequence_length=5,
    batch_size=64,
    epochs=10,
    lr=0.001
):
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # -----------------------------
    # Scale features
    # -----------------------------
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # -----------------------------
    # Create sequences
    # -----------------------------
    X, y = create_sequences(df, feature_cols, target_col, sequence_length)

    print(f"Created {len(X)} sequences of length {sequence_length}")

    # -----------------------------
    # Train/Test split
    # -----------------------------
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_dataset = InjuryDataset(X_train, y_train)
    test_dataset = InjuryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # -----------------------------
    # Model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InjuryLSTM(input_size=len(feature_cols)).to(device)

    pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Training loop
    # -----------------------------
    print("Training RNN...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()
    y_probs = []
    y_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()

            y_probs.extend(outputs)
            y_true.extend(y_batch.numpy())

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    threshold = 0.01
    y_pred = (y_probs >= threshold).astype(int)

    print("\n=== RNN Evaluation ===")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_probs))

    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]

    print("Best Threshold:", best_threshold)
    return {
        "model": model,
        "y_probs": y_probs,
        "y_pred": y_pred
    }