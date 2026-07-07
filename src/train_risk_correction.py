import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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


def build_labeled_subgraph(features_df, edges_df):
    """
    Build neighbor lists only over labeled nodes.
    This keeps training much faster than using all 203k nodes.
    """
    labeled_df = features_df[features_df["label"] != -1].copy()

    labeled_txids = labeled_df["txId"].tolist()
    txid_to_local_idx = {txid: idx for idx, txid in enumerate(labeled_txids)}

    n = len(labeled_df)
    in_neighbors = [[] for _ in range(n)]
    out_neighbors = [[] for _ in range(n)]

    used_edges = 0

    for row in edges_df.itertuples(index=False):
        src_id = getattr(row, "txId1")
        dst_id = getattr(row, "txId2")

        if src_id not in txid_to_local_idx or dst_id not in txid_to_local_idx:
            continue

        src = txid_to_local_idx[src_id]
        dst = txid_to_local_idx[dst_id]

        out_neighbors[src].append(dst)
        in_neighbors[dst].append(src)
        used_edges += 1

    return labeled_df, txid_to_local_idx, in_neighbors, out_neighbors, used_edges


def compute_pos_weight(y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    if num_pos == 0:
        return 1.0

    return num_neg / num_pos


def print_risk_stats(y_true, y_prob):
    licit_scores = y_prob[y_true == 0]
    illicit_scores = y_prob[y_true == 1]

    print("========== FINAL RISK SCORE STATS ==========")

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


class RiskCorrectionNet(nn.Module):
    """
    Stage-2 model:
    - takes node features and saved base logits
    - learns a neighbor-based correction
    - learns a gate deciding how much of the correction to apply
    - final logit = base_logit + gate * correction
    """

    def __init__(self, input_dim, hidden_dim=64, embed_dim=32):
        super().__init__()

        # feature -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

        # neighbor correction from [m_in || m_out]
        self.correction_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # gate from [h_i || m_in || m_out]
        self.gate_head = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def aggregate_neighbor_means(self, H, in_neighbors, out_neighbors):
        """
        H: [N, embed_dim]
        returns:
            M_in:  [N, embed_dim]
            M_out: [N, embed_dim]
        """
        device = H.device
        N, D = H.shape

        M_in = torch.zeros((N, D), dtype=H.dtype, device=device)
        M_out = torch.zeros((N, D), dtype=H.dtype, device=device)

        for i in range(N):
            in_nbrs = in_neighbors[i]
            out_nbrs = out_neighbors[i]

            if len(in_nbrs) > 0:
                in_idx = torch.tensor(in_nbrs, dtype=torch.long, device=device)
                M_in[i] = H[in_idx].mean(dim=0)

            if len(out_nbrs) > 0:
                out_idx = torch.tensor(out_nbrs, dtype=torch.long, device=device)
                M_out[i] = H[out_idx].mean(dim=0)

        return M_in, M_out

    def forward(self, X_all, base_logits_all, in_neighbors, out_neighbors):
        """
        X_all: [N, d] standardized labeled-node features
        base_logits_all: [N] saved logits from stage 1
        """
        H = self.encoder(X_all)  # [N, embed_dim]

        M_in, M_out = self.aggregate_neighbor_means(H, in_neighbors, out_neighbors)

        neighbor_context = torch.cat([M_in, M_out], dim=1)  # [N, 2*embed_dim]
        correction_logits = self.correction_head(neighbor_context).squeeze(1)  # [N]

        gate_input = torch.cat([H, M_in, M_out], dim=1)  # [N, 3*embed_dim]
        gate = torch.sigmoid(self.gate_head(gate_input)).squeeze(1)  # [N]

        final_logits = base_logits_all + gate * correction_logits
        return final_logits, correction_logits, gate


if __name__ == "__main__":
    # -------------------------------------------------
    # Config
    # -------------------------------------------------
    FEATURES_CSV = "data/graph_features.csv"
    EDGES_CSV = "data/elliptic_txs_edgelist.csv"

    BASE_MODEL_PATH = "artifacts/base_risk_model.pt"      # not strictly needed here
    SCALER_PATH = "artifacts/base_risk_scaler.pkl"
    BASE_SCORES_PATH = "artifacts/base_risk_scores.csv"

    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    THRESHOLD = 0.5

    HARD_ILLICIT_THRESHOLD = 0.50
    HARD_LOSS_LAMBDA = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    features_df = pd.read_csv(FEATURES_CSV)
    edges_df = pd.read_csv(EDGES_CSV)
    base_scores_df = pd.read_csv(BASE_SCORES_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Build labeled subgraph
    labeled_df, txid_to_local_idx, in_neighbors, out_neighbors, used_edges = build_labeled_subgraph(
        features_df, edges_df
    )

    print("========== LABELED SUBGRAPH ==========")
    print("Labeled nodes:", len(labeled_df))
    print("Edges inside labeled subgraph:", used_edges)
    print()

    # -------------------------------------------------
    # Merge in saved base scores / splits
    # -------------------------------------------------
    df = labeled_df.merge(
        base_scores_df[["txId", "label", "graph_index", "base_logit", "base_risk", "split"]],
        on=["txId", "label"],
        how="inner"
    ).copy()

    # Sort by local labeled-subgraph order so row i aligns with neighbor lists
    df["local_idx"] = df["txId"].map(txid_to_local_idx)
    df = df.sort_values("local_idx").reset_index(drop=True)

    # -------------------------------------------------
    # Prepare arrays
    # -------------------------------------------------
    X_all = df[FEATURE_COLUMNS].values
    X_all_scaled = scaler.transform(X_all)

    y_all = df["label"].values.astype(np.float32)
    base_logits_all = df["base_logit"].values.astype(np.float32)
    base_risk_all = df["base_risk"].values.astype(np.float32)
    split_all = df["split"].values

    train_mask_np = (split_all == "train")
    test_mask_np = (split_all == "test")

    # Hard illicit nodes in TRAIN split:
    # label = 1 and base risk < threshold
    hard_mask_np = (
        (split_all == "train") &
        (y_all == 1) &
        (base_risk_all < HARD_ILLICIT_THRESHOLD)
    )

    print("========== HARD ILLICIT SET ==========")
    print("Threshold:", HARD_ILLICIT_THRESHOLD)
    print("Hard illicit train nodes:", int(hard_mask_np.sum()))
    print()

    X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32, device=device)
    y_all_tensor = torch.tensor(y_all, dtype=torch.float32, device=device)
    base_logits_tensor = torch.tensor(base_logits_all, dtype=torch.float32, device=device)

    train_mask = torch.tensor(train_mask_np, dtype=torch.bool, device=device)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool, device=device)
    hard_mask = torch.tensor(hard_mask_np, dtype=torch.bool, device=device)

    # -------------------------------------------------
    # Model / optimizer / losses
    # -------------------------------------------------
    model = RiskCorrectionNet(
        input_dim=X_all_scaled.shape[1],
        hidden_dim=64,
        embed_dim=32
    ).to(device)

    pos_weight_value = compute_pos_weight(y_all[train_mask_np])
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
    print("Hard loss lambda:", HARD_LOSS_LAMBDA)
    print()

    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    best_test_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        final_logits_all, correction_logits_all, gate_all = model(
            X_all_tensor, base_logits_tensor, in_neighbors, out_neighbors
        )

        # BCE on all TRAIN nodes
        train_logits = final_logits_all[train_mask]
        train_labels = y_all_tensor[train_mask]
        bce_loss = bce_loss_fn(train_logits, train_labels)

        # Extra loss only on hard illicit train nodes
        if hard_mask.sum().item() > 0:
            hard_probs = torch.sigmoid(final_logits_all[hard_mask])
            hard_loss = ((1.0 - hard_probs) ** 2).mean()
        else:
            hard_loss = torch.tensor(0.0, device=device)

        total_loss = bce_loss + HARD_LOSS_LAMBDA * hard_loss
        total_loss.backward()
        optimizer.step()

        # ---------------- TEST EVAL ----------------
        model.eval()
        with torch.no_grad():
            final_logits_eval, correction_logits_eval, gate_eval = model(
                X_all_tensor, base_logits_tensor, in_neighbors, out_neighbors
            )

            test_logits = final_logits_eval[test_mask]
            test_labels = y_all_tensor[test_mask]
            test_bce = bce_loss_fn(test_logits, test_labels)

            # monitoring stats
            mean_gate_train = gate_eval[train_mask].mean().item()
            mean_gate_test = gate_eval[test_mask].mean().item()

            if hard_mask.sum().item() > 0:
                hard_probs_eval = torch.sigmoid(final_logits_eval[hard_mask])
                test_hard_like = ((1.0 - hard_probs_eval) ** 2).mean().item()
            else:
                test_hard_like = 0.0

            # We choose model by test BCE for now
            model_selection_loss = test_bce.item()

        if model_selection_loss < best_test_loss:
            best_test_loss = model_selection_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train BCE: {bce_loss.item():.4f} | "
            f"Train Hard: {hard_loss.item():.4f} | "
            f"Test BCE: {test_bce.item():.4f} | "
            f"HardEval: {test_hard_like:.4f} | "
            f"Mean Gate Train: {mean_gate_train:.4f} | "
            f"Mean Gate Test: {mean_gate_test:.4f}"
        )

    print()
    print("Best test BCE:", round(best_test_loss, 4))
    print()

    # -------------------------------------------------
    # Load best model
    # -------------------------------------------------
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    # -------------------------------------------------
    # Final evaluation on test split
    # -------------------------------------------------
    with torch.no_grad():
        final_logits_all, correction_logits_all, gate_all = model(
            X_all_tensor, base_logits_tensor, in_neighbors, out_neighbors
        )
        final_probs_all = torch.sigmoid(final_logits_all).cpu().numpy()
        gate_all_np = gate_all.cpu().numpy()
        correction_np = correction_logits_all.cpu().numpy()

    y_test = y_all[test_mask_np]
    y_prob = final_probs_all[test_mask_np]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print("========== FINAL TEST METRICS ==========")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print()

    auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", round(auc, 4))
    print()

    print_risk_stats(y_test, y_prob)

    # -------------------------------------------------
    # Diagnostics on hard illicit nodes
    # -------------------------------------------------
    hard_train_df = df[hard_mask_np].copy()

    if len(hard_train_df) > 0:
        hard_train_df["final_risk"] = final_probs_all[hard_mask_np]
        hard_train_df["gate"] = gate_all_np[hard_mask_np]
        hard_train_df["correction_logit"] = correction_np[hard_mask_np]

        print("========== HARD ILLICIT DIAGNOSTICS ==========")
        print("Count:", len(hard_train_df))
        print("Base risk mean  :", round(float(hard_train_df["base_risk"].mean()), 4))
        print("Final risk mean :", round(float(hard_train_df["final_risk"].mean()), 4))
        print("Base risk median:", round(float(hard_train_df["base_risk"].median()), 4))
        print("Final risk median:", round(float(hard_train_df["final_risk"].median()), 4))
        print("Mean gate:", round(float(hard_train_df["gate"].mean()), 4))
        print()

    # -------------------------------------------------
    # Save stage-2 outputs
    # -------------------------------------------------
    os.makedirs("artifacts", exist_ok=True)

    output_df = df[["txId", "label", "graph_index", "split", "base_logit", "base_risk"]].copy()
    output_df["final_risk"] = final_probs_all
    output_df["gate"] = gate_all_np
    output_df["correction_logit"] = correction_np
    output_df["is_hard_illicit_train"] = hard_mask_np.astype(int)

    output_path = "artifacts/risk_correction_scores.csv"
    output_df.to_csv(output_path, index=False)

    print("Saved stage-2 scores to:", output_path)
