from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from load_elliptic_raw import load_elliptic_raw


FEATURE_COLUMNS = [
    "indegree",
    "outdegree",
    "degree_imbalance",
    "pass_through",
    "split_score",
    "merge_score",
    "pagerank",
    "hub_score",
    "authority_score",
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
    "bridge_relay_score",
    "bridge_cross_density",
    "bridge_score",
]

RELAY_FEATURE_COLUMNS = [
    "pass_through",
    "degree_imbalance",
    "split_score",
    "merge_score",
    "temporal_pass_through",
    "bridge_score",
    "core_number",
]


def build_elliptic_gnn_data(
    features_csv_path="data/graph_features.csv",
    classes_path="data/elliptic_txs_classes.csv",
    edgelist_path="data/elliptic_txs_edgelist.csv",
    features_path="data/elliptic_txs_features.csv",
    test_size=0.2,
    random_state=42,
):
    features_df = pd.read_csv(features_csv_path)

    raw_data = load_elliptic_raw(
        classes_path=classes_path,
        edgelist_path=edgelist_path,
        features_path=features_path
    )

    features_df = features_df.sort_values("txId").reset_index(drop=True)

    tx_ids = features_df["txId"].tolist()
    node_to_idx = {}
    for idx, tx_id in enumerate(tx_ids):
        node_to_idx[tx_id] = idx

    x = torch.tensor(features_df[FEATURE_COLUMNS].values, dtype=torch.float32)
    relay_x = torch.tensor(features_df[RELAY_FEATURE_COLUMNS].values, dtype=torch.float32)

    labels = features_df["label"].astype(int).tolist()
    y = torch.tensor(labels, dtype=torch.long)

    timesteps = torch.tensor(features_df["timestep"].astype(int).values, dtype=torch.long)
    tx_id_tensor = torch.tensor(features_df["txId"].astype(int).values, dtype=torch.long)

    edge_src = []
    edge_dst = []

    for src, dst in raw_data.graph.edges():
        if src in node_to_idx and dst in node_to_idx:
            edge_src.append(node_to_idx[src])
            edge_dst.append(node_to_idx[dst])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    num_nodes = len(features_df)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    labeled_indices = []
    labeled_targets = []

    for idx, label in enumerate(labels):
        if label != -1:
            labeled_indices.append(idx)
            labeled_targets.append(label)

    train_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labeled_targets
    )

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    data.relay_x = relay_x
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.timesteps = timesteps
    data.tx_ids = tx_id_tensor
    data.feature_columns = FEATURE_COLUMNS
    data.relay_feature_columns = RELAY_FEATURE_COLUMNS

    return data


if __name__ == "__main__":
    data = build_elliptic_gnn_data()

    print("========== GNN DATA ==========")
    print("Num nodes:", data.num_nodes)
    print("Num edges:", data.edge_index.size(1))
    print("Feature dim:", data.x.size(1))
    print("Relay feature dim:", data.relay_x.size(1))
    print()

    labeled_mask = data.y != -1
    num_labeled = int(labeled_mask.sum().item())
    num_licit = int((data.y == 0).sum().item())
    num_illicit = int((data.y == 1).sum().item())

    print("Labeled nodes:", num_labeled)
    print("Licit nodes:", num_licit)
    print("Illicit nodes:", num_illicit)
    print()

    print("Train nodes:", int(data.train_mask.sum().item()))
    print("Test nodes:", int(data.test_mask.sum().item()))
    print()

    print("Train illicit:", int(data.y[data.train_mask].sum().item()))
    print("Test illicit:", int(data.y[data.test_mask].sum().item()))
