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
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, auc
)
from imblearn.combine import SMOTEENN  # Handles class imbalance using oversampling + cleaning

# -----------------------------
# 1️⃣ Sequence Dataset
# -----------------------------
class InjuryDataset(Dataset):
    """
    Custom PyTorch Dataset for sequential injury prediction.
    Each sample is a sequence of features over 'sequence_length' games,
    with a corresponding target (injury event in the next game).
    """
    def __init__(self, X, y):
        # Convert data to torch tensors with appropriate dtype
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)  # Number of sequences

    def __getitem__(self, idx):
        # Returns a single sequence and its corresponding label
        return self.X[idx], self.y[idx]


# -----------------------------
# 2️⃣ LSTM Model
# -----------------------------
class InjuryLSTM(nn.Module):
    """
    Simple stacked LSTM model for sequence-to-one prediction.
    Takes a sequence of feature vectors and outputs a single logit
    representing the likelihood of injury in the next game.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        # LSTM layer: batch_first=True allows input shape [batch, seq, features]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer reduces hidden state to single output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Only use the hidden state of the last time step (hn[-1])
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out  # raw logits (not probabilities yet)


# -----------------------------
# 3️⃣ Create Sequences
# -----------------------------
def create_sequences(df, feature_cols, target_col, sequence_length=5):
    """
    Converts a dataframe into sequences suitable for RNN input.
    - df: pandas DataFrame containing game-level features
    - feature_cols: list of columns used as input features
    - target_col: column used as the prediction target (injury)
    - sequence_length: number of consecutive games per input sequence

    Returns:
    - X: array of sequences [num_sequences, seq_len, num_features]
    - y: array of labels [num_sequences]
    """
    sequences = []
    labels = []

    # Sort by player and date to preserve temporal order
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    for player_id, group in df.groupby("PLAYER_ID"):
        group = group.reset_index(drop=True)
        for i in range(len(group) - sequence_length):
            seq = group[feature_cols].iloc[i:i+sequence_length].values
            label = group[target_col].iloc[i+sequence_length]  # Next game injury
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
    epochs=20,
    lr=0.001,
    use_smoteenn=True
):
    """
    Full training pipeline for LSTM injury prediction.
    Handles preprocessing, sequence creation, rare-event resampling, training, and evaluation.
    """

    print("Loading data...")
    df = pd.read_csv(csv_path)

    # -----------------------------
    # Feature scaling
    # -----------------------------
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    # Standardization is critical for LSTM training to ensure gradients don't explode/vanish

    # -----------------------------
    # Create sequences
    # -----------------------------
    X, y = create_sequences(df, feature_cols, target_col, sequence_length)
    print(f"Created {len(X)} sequences of length {sequence_length}")

    # -----------------------------
    # Train/Test split (80/20)
    # -----------------------------
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -----------------------------
    # Flatten sequences for resampling
    # -----------------------------
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    # -----------------------------
    # Handle class imbalance (rare injuries)
    # -----------------------------
    if use_smoteenn:
        # SMOTE + ENN reduces bias from oversampling by cleaning overlapping samples
        sampler = SMOTEENN(random_state=42)
        X_train_res, y_train_res = sampler.fit_resample(X_train_flat, y_train)
    else:
        # Basic SMOTE only
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=42)
        X_train_res, y_train_res = sampler.fit_resample(X_train_flat, y_train)

    # Reshape back to sequence format for LSTM
    X_train_res = X_train_res.reshape(-1, sequence_length, len(feature_cols))
    print(f"After resampling, training dataset size: {X_train_res.shape[0]}")

    # -----------------------------
    # Datasets & DataLoaders
    # -----------------------------
    train_dataset = InjuryDataset(X_train_res, y_train_res)
    test_dataset = InjuryDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # -----------------------------
    # Device setup (GPU if available)
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used for torch:", device)

    model = InjuryLSTM(input_size=len(feature_cols)).to(device)

    # -----------------------------
    # Weighted loss for rare-event class
    # -----------------------------
    num_pos = np.sum(y_train_res == 1)
    num_neg = np.sum(y_train_res == 0)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Weighting helps prevent the model from always predicting the majority class

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Training loop
    # -----------------------------
    print("Training RNN...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()
    y_logits, y_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze().cpu()
            y_logits.extend(logits.numpy())
            y_true.extend(y_batch.numpy())

    y_logits = np.array(y_logits)
    y_true = np.array(y_true)

    # Convert logits to probabilities using sigmoid
    y_probs = 1 / (1 + np.exp(-y_logits))

    # -----------------------------
    # Find threshold that maximizes F1 score
    # -----------------------------
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]

    # Predictions using optimal threshold
    y_pred = (y_probs >= best_threshold).astype(int)

    print("\n=== RNN Evaluation ===")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    pr_auc = auc(recall, precision)
    print("PR-AUC:", pr_auc)
    print("Best Threshold (F1-max):", best_threshold)

    # -----------------------------
    # Return model and evaluation results for further analysis
    # -----------------------------
    return {
        "model": model,
        "y_probs": y_probs,
        "y_pred": y_pred,
        "best_threshold": best_threshold
    }