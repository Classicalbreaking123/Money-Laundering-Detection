import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


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


class RiskMLP(nn.Module):
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


def load_full_graph_data(features_csv_path, edges_csv_path):
    features_df = pd.read_csv(features_csv_path)
    edges_df = pd.read_csv(edges_csv_path)
    return features_df, edges_df


def build_graph_mappings(features_df, edges_df):
    node_ids = features_df["txId"].tolist()
    id_to_idx = {txid: idx for idx, txid in enumerate(node_ids)}
    n = len(node_ids)

    in_neighbors = [[] for _ in range(n)]
    out_neighbors = [[] for _ in range(n)]

    valid_edge_count = 0

    for row in edges_df.itertuples(index=False):
        src_id = getattr(row, "txId1")
        dst_id = getattr(row, "txId2")

        if src_id not in id_to_idx or dst_id not in id_to_idx:
            continue

        src = id_to_idx[src_id]
        dst = id_to_idx[dst_id]

        out_neighbors[src].append(dst)
        in_neighbors[dst].append(src)
        valid_edge_count += 1

    return id_to_idx, in_neighbors, out_neighbors, valid_edge_count


def compute_pos_weight(y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    if num_pos == 0:
        return 1.0

    return num_neg / num_pos


def compute_flow_loss_detached(risk_scores, node_indices, in_neighbors, out_neighbors, device):
    """
    Detached-neighbor geometric-mean flow loss.

    For node i:
        R_in  = mean detached risk of incoming neighbors
        R_out = mean detached risk of outgoing neighbors
        F_i   = sqrt(R_in * R_out)

        loss_i = (r_i - F_i)^2

    Only nodes with at least one incoming and one outgoing neighbor are included.
    """
    losses = []
    risk_scores_detached = risk_scores.detach()

    for i in node_indices:
        in_nbrs = in_neighbors[i]
        out_nbrs = out_neighbors[i]

        if len(in_nbrs) == 0 or len(out_nbrs) == 0:
            continue

        in_idx = torch.tensor(in_nbrs, dtype=torch.long, device=device)
        out_idx = torch.tensor(out_nbrs, dtype=torch.long, device=device)

        R_in = risk_scores_detached[in_idx].mean()
        R_out = risk_scores_detached[out_idx].mean()

        flow_target = 0.5*(R_in * R_out )
        losses.append((risk_scores[i] - flow_target) ** 2)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def compute_flow_stats_detached(risk_scores, node_indices, in_neighbors, out_neighbors, device):
    """
    Diagnostics:
    - mean flow target
    - mean |risk - flow_target|
    """
    flow_targets = []
    abs_gaps = []

    risk_scores_detached = risk_scores.detach()

    for i in node_indices:
        in_nbrs = in_neighbors[i]
        out_nbrs = out_neighbors[i]

        if len(in_nbrs) == 0 or len(out_nbrs) == 0:
            continue

        in_idx = torch.tensor(in_nbrs, dtype=torch.long, device=device)
        out_idx = torch.tensor(out_nbrs, dtype=torch.long, device=device)

        R_in = risk_scores_detached[in_idx].mean()
        R_out = risk_scores_detached[out_idx].mean()
        flow_target =0.5*(R_in * R_out )

        flow_targets.append(flow_target)
        abs_gaps.append(torch.abs(risk_scores[i] - flow_target))

    if len(flow_targets) == 0:
        return 0.0, 0.0

    mean_target = torch.stack(flow_targets).mean().item()
    mean_gap = torch.stack(abs_gaps).mean().item()
    return mean_target, mean_gap


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
    EDGES_CSV = "data/elliptic_txs_edgelist.csv"

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 675

    THRESHOLD = 0.5

    # Risk propagation regularization strength
    FLOW_LAMBDA = 0.1

    # Checkpoint / artifacts
    CHECKPOINT_DIR = "artifacts"
    LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "baseline_latest.pt")
    BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "baseline_best.pt")
    SCALER_PATH = os.path.join(CHECKPOINT_DIR, "baseline_scaler.pkl")
    SCORES_PATH = os.path.join(CHECKPOINT_DIR, "baseline_risk_scores.csv")

    RESUME_TRAINING = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -------------------------------------------------
    # Load full graph data
    # -------------------------------------------------
    features_df, edges_df = load_full_graph_data(FEATURES_CSV, EDGES_CSV)

    print("========== FULL GRAPH DATA ==========")
    print("Nodes:", len(features_df))
    print("Edges (raw):", len(edges_df))
    print()

    id_to_idx, in_neighbors, out_neighbors, valid_edge_count = build_graph_mappings(
        features_df, edges_df
    )

    print("Edges used after filtering to known nodes:", valid_edge_count)
    print()

    # -------------------------------------------------
    # Labeled subset
    # -------------------------------------------------
    labeled_df = features_df[features_df["label"] != -1].copy()

    print("========== LABELED DATA ==========")
    print("Shape:", labeled_df.shape)
    print()
    print("Label counts:")
    print(labeled_df["label"].value_counts().sort_index())
    print()

    labeled_indices = labeled_df.index.to_numpy()
    y_all_labeled = labeled_df["label"].values.astype(np.float32)

    train_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all_labeled
    )

    train_df = features_df.loc[train_idx].copy()
    test_df = features_df.loc[test_idx].copy()

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label"].values.astype(np.float32)

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values.astype(np.float32)

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print()

    # -------------------------------------------------
    # Standardize using TRAIN labeled nodes only
    # -------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_PATH)

    X_all = features_df[FEATURE_COLUMNS].values
    X_all_scaled = scaler.transform(X_all)

    X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

    train_graph_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    test_graph_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

    # -------------------------------------------------
    # Model / loss / optimizer
    # -------------------------------------------------
    model = RiskMLP(input_dim=X_all_scaled.shape[1]).to(device)

    pos_weight_value = compute_pos_weight(y_train)
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    print("========== TRAINING CONFIG ==========")
    print("Device:", device)
    print("Input dim:", X_all_scaled.shape[1])
    print("Positive class weight:", round(pos_weight_value, 4))
    print("Flow lambda:", FLOW_LAMBDA)
    print("Flow target: sqrt(R_in * R_out) from DETACHED neighbor risks")
    print("Flow penalty: (risk - flow_target)^2")
    print("Resume training:", RESUME_TRAINING)
    print()

    # -------------------------------------------------
    # Resume if checkpoint exists
    # -------------------------------------------------
    start_epoch = 1
    best_test_total_loss = float("inf")
    best_state_dict = None

    if RESUME_TRAINING and os.path.exists(LATEST_CKPT_PATH):
        print("========== RESUMING FROM CHECKPOINT ==========")
        checkpoint = torch.load(LATEST_CKPT_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_test_total_loss = checkpoint["best_test_total_loss"]

        if "best_model_state_dict" in checkpoint and checkpoint["best_model_state_dict"] is not None:
            best_state_dict = checkpoint["best_model_state_dict"]

        print("Resumed from epoch:", checkpoint["epoch"])
        print("Best test total loss so far:", round(best_test_total_loss, 4))
        print()

    if start_epoch > EPOCHS:
        print("Checkpoint already reached or exceeded requested EPOCHS.")
        print("No further training needed.")
        print()

    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits_all = model(X_all_tensor)
        probs_all = torch.sigmoid(logits_all)

        # BCE only on TRAIN labeled nodes
        train_logits = logits_all[train_graph_idx]
        cls_loss = bce_loss_fn(train_logits, y_train_tensor)

        # Risk propagation / flow loss on TRAIN nodes
        flow_loss = compute_flow_loss_detached(
            risk_scores=probs_all,
            node_indices=train_idx,
            in_neighbors=in_neighbors,
            out_neighbors=out_neighbors,
            device=device
        )

        total_loss = cls_loss + FLOW_LAMBDA * flow_loss
        total_loss.backward()
        optimizer.step()

        # ---------------- TEST EVAL ----------------
        model.eval()
        with torch.no_grad():
            logits_all_eval = model(X_all_tensor)
            probs_all_eval = torch.sigmoid(logits_all_eval)

            test_logits = logits_all_eval[test_graph_idx]
            test_cls_loss = bce_loss_fn(test_logits, y_test_tensor)

            test_flow_loss = compute_flow_loss_detached(
                risk_scores=probs_all_eval,
                node_indices=test_idx,
                in_neighbors=in_neighbors,
                out_neighbors=out_neighbors,
                device=device
            )

            test_total_loss = test_cls_loss + FLOW_LAMBDA * test_flow_loss

            train_mean_target, train_mean_gap = compute_flow_stats_detached(
                risk_scores=probs_all_eval,
                node_indices=train_idx,
                in_neighbors=in_neighbors,
                out_neighbors=out_neighbors,
                device=device
            )

            test_mean_target, test_mean_gap = compute_flow_stats_detached(
                risk_scores=probs_all_eval,
                node_indices=test_idx,
                in_neighbors=in_neighbors,
                out_neighbors=out_neighbors,
                device=device
            )

        # Save best checkpoint
        if test_total_loss.item() < best_test_total_loss:
            best_test_total_loss = test_total_loss.item()
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_test_total_loss": best_test_total_loss,
                    "best_model_state_dict": best_state_dict,
                },
                BEST_CKPT_PATH
            )

        # Save latest checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_total_loss": best_test_total_loss,
                "best_model_state_dict": best_state_dict,
            },
            LATEST_CKPT_PATH
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Total: {total_loss.item():.4f} | "
            f"Train BCE: {cls_loss.item():.4f} | "
            f"Train Flow: {flow_loss.item():.4f} | "
            f"Test Total: {test_total_loss.item():.4f} | "
            f"Test BCE: {test_cls_loss.item():.4f} | "
            f"Test Flow: {test_flow_loss.item():.4f} | "
            f"Train |r-F|: {train_mean_gap:.4f} | "
            f"Test |r-F|: {test_mean_gap:.4f}"
        )

    print()
    print("Best test total loss:", round(best_test_total_loss, 4))
    print()

    # -------------------------------------------------
    # Load best model
    # -------------------------------------------------
    if best_state_dict is None:
        if os.path.exists(BEST_CKPT_PATH):
            best_checkpoint = torch.load(BEST_CKPT_PATH, map_location=device)
            best_state_dict = best_checkpoint["best_model_state_dict"]
        else:
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    # -------------------------------------------------
    # Final predictions on all nodes
    # -------------------------------------------------
    with torch.no_grad():
        logits_all = model(X_all_tensor)
        probs_all = torch.sigmoid(logits_all).cpu().numpy()
        logits_all_np = logits_all.cpu().numpy()

    y_prob = probs_all[test_idx]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    print("========== CONFUSION MATRIX ==========")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("========== CLASSIFICATION REPORT ==========")
    print(classification_report(y_test, y_pred, digits=4))
    print()

    auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", round(auc, 4))
    print()

    # -------------------------------------------------
    # Risk stats
    # -------------------------------------------------
    print_risk_stats(y_test, y_prob)

    # -------------------------------------------------
    # Save risk scores CSV
    # -------------------------------------------------
    out_df = features_df[["txId", "label"]].copy()
    out_df["risk_score"] = probs_all
    out_df["logit"] = logits_all_np
    out_df["split"] = "unlabeled"

    out_df.loc[train_idx, "split"] = "train"
    out_df.loc[test_idx, "split"] = "test"

    out_df.to_csv(SCORES_PATH, index=False)

    print("========== SAVED ARTIFACTS ==========")
    print("Latest checkpoint :", LATEST_CKPT_PATH)
    print("Best checkpoint   :", BEST_CKPT_PATH)
    print("Scaler            :", SCALER_PATH)
    print("Scores CSV        :", SCORES_PATH)
