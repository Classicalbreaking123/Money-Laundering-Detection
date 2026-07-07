import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


FEATURE_COLUMNS = [
    "indegree",
    "outdegree",
    "degree_imbalance",
    "pass_through",
    "split_score",
    "merge_score",
    "pagerank",
    "scc_size",
    "wcc_size",
    "cycle_flag",
    "core_number",
    "two_hop_reach",
    "avg_neighbor_degree",
    "temporal_burst",
    "temporal_in_degree",
    "temporal_out_degree",
    "temporal_pass_through",
    "clustering_coeff",
    "hub_score",
    "authority_score",
    "bridge_relay_score",
    "bridge_cross_density",
    "bridge_score",
]


class RiskDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BaseRiskMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        logits = self.net(x).squeeze(1)
        return logits


def load_labeled_data(features_csv_path):
    df = pd.read_csv(features_csv_path)
    df = df[df["label"] != -1].copy()

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values.astype(np.float32)

    return df, X, y


def compute_pos_weight(y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    if num_pos == 0:
        return 1.0

    return num_neg / num_pos


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_logits_and_probs(model, X, device, batch_size=1024):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_probs = []

    for X_batch in loader:
        X_batch = X_batch.to(device)

        logits = model(X_batch)
        probs = torch.sigmoid(logits)

        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_logits), np.concatenate(all_probs)


def print_risk_stats(y_true, y_prob):
    licit_scores = y_prob[y_true == 0]
    illicit_scores = y_prob[y_true == 1]

    print("========== RISK SCORE STATS ==========")

    if len(licit_scores) > 0:
        print("LICIT NODES")
        print("  Count  :", len(licit_scores))
        print("  Mean   :", round(float(np.mean(licit_scores)), 4))
        print("  Median :", round(float(np.median(licit_scores)), 4))
        print("  Std    :", round(float(np.std(licit_scores)), 4))
        print("  Min    :", round(float(np.min(licit_scores)), 4))
        print("  Max    :", round(float(np.max(licit_scores)), 4))
        print()

    if len(illicit_scores) > 0:
        print("ILLICIT NODES")
        print("  Count  :", len(illicit_scores))
        print("  Mean   :", round(float(np.mean(illicit_scores)), 4))
        print("  Median :", round(float(np.median(illicit_scores)), 4))
        print("  Std    :", round(float(np.std(illicit_scores)), 4))
        print("  Min    :", round(float(np.min(illicit_scores)), 4))
        print("  Max    :", round(float(np.max(illicit_scores)), 4))
        print()

    print("Percentiles:")
    for cls_name, scores in [("Licit", licit_scores), ("Illicit", illicit_scores)]:
        if len(scores) == 0:
            continue

        p25 = np.percentile(scores, 25)
        p50 = np.percentile(scores, 50)
        p75 = np.percentile(scores, 75)
        p90 = np.percentile(scores, 90)

        print(
            f"  {cls_name:<7} -> "
            f"P25={p25:.4f}, P50={p50:.4f}, P75={p75:.4f}, P90={p90:.4f}"
        )
    print()


if __name__ == "__main__":
    # -------------------------------------------------
    # Config
    # -------------------------------------------------
    FEATURES_CSV = "data/graph_features.csv"

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    BATCH_SIZE = 256
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 40

    THRESHOLD = 0.5

    MODEL_SAVE_PATH = "artifacts/base_risk_model.pt"
    SCALER_SAVE_PATH = "artifacts/base_risk_scaler.pkl"
    SCORES_SAVE_PATH = "artifacts/base_risk_scores.csv"
    SPLIT_SAVE_PATH = "artifacts/base_risk_split.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Load labeled data
    # -------------------------------------------------
    df, X, y = load_labeled_data(FEATURES_CSV)

    print("========== LABELED DATA ==========")
    print("Shape:", df.shape)
    print()
    print("Label counts:")
    print(pd.Series(y).value_counts().sort_index())
    print()

    # Keep original row indices so we can later map back into the full graph
    all_indices = df.index.to_numpy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print()

    # -------------------------------------------------
    # Standardize
    # -------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Also transform ALL labeled nodes so we can save base scores for everyone
    X_all_scaled = scaler.transform(X)

    # -------------------------------------------------
    # Dataloaders
    # -------------------------------------------------
    train_dataset = RiskDataset(X_train, y_train)
    test_dataset = RiskDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------------------------------
    # Model / loss / optimizer
    # -------------------------------------------------
    model = BaseRiskMLP(input_dim=X_train.shape[1]).to(device)

    pos_weight_value = compute_pos_weight(y_train)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    print("========== TRAINING CONFIG ==========")
    print("Device:", device)
    print("Input dim:", X_train.shape[1])
    print("Positive class weight:", round(pos_weight_value, 4))
    print()

    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    best_test_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate_loss(model, test_loader, criterion, device)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train BCE: {train_loss:.4f} | "
            f"Test BCE: {test_loss:.4f}"
        )

    print()
    print("Best test BCE:", round(best_test_loss, 4))
    print()

    # -------------------------------------------------
    # Load best model
    # -------------------------------------------------
    model.load_state_dict(best_state_dict)
    model.to(device)

    # -------------------------------------------------
    # Test metrics
    # -------------------------------------------------
    test_logits, test_probs = predict_logits_and_probs(model, X_test, device)
    test_pred = (test_probs >= THRESHOLD).astype(int)

    print("========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y_test, test_pred))
    print()

    print("========== CLASSIFICATION REPORT ==========")
    print(classification_report(y_test, test_pred, digits=4))
    print()

    auc = roc_auc_score(y_test, test_probs)
    print("ROC-AUC:", round(auc, 4))
    print()

    print_risk_stats(y_test, test_probs)

    # -------------------------------------------------
    # Save base scores for ALL labeled nodes
    # -------------------------------------------------
    all_logits, all_probs = predict_logits_and_probs(model, X_all_scaled, device)

    scores_df = df[["txId", "label"]].copy()
    scores_df["graph_index"] = df.index
    scores_df["base_logit"] = all_logits
    scores_df["base_risk"] = all_probs

    # Mark whether each labeled node belonged to train or test split
    split_map = {}
    for idx in idx_train:
        split_map[idx] = "train"
    for idx in idx_test:
        split_map[idx] = "test"

    scores_df["split"] = scores_df["graph_index"].map(split_map)

    # Save a compact split file too
    split_df = scores_df[["txId", "graph_index", "label", "split"]].copy()

    # Make sure artifacts directory exists
    import os
    os.makedirs("artifacts", exist_ok=True)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    scores_df.to_csv(SCORES_SAVE_PATH, index=False)
    split_df.to_csv(SPLIT_SAVE_PATH, index=False)

    print("========== SAVED ARTIFACTS ==========")
    print("Model  :", MODEL_SAVE_PATH)
    print("Scaler :", SCALER_SAVE_PATH)
    print("Scores :", SCORES_SAVE_PATH)
    print("Split  :", SPLIT_SAVE_PATH)
    print()
