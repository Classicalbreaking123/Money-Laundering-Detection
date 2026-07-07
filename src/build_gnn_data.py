import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


RELAY_FEATURE_COLUMNS = [
    "indegree",
    "outdegree",
    "pass_through",
    "split_score",
    "merge_score",
    "degree_imbalance",
    "temporal_in_degree",
    "temporal_out_degree",
    "temporal_pass_through",
    "hub_score",
    "authority_score",
    "bridge_relay_score",
    "bridge_cross_density",
    "bridge_score",
]


def get_node_feature_columns(df):
    exclude_cols = {"txId", "timestep", "label"}
    feature_cols = []

    for col in df.columns:
        if col not in exclude_cols:
            feature_cols.append(col)

    return feature_cols


def build_edge_index(edges_csv_path, txid_to_idx):
    edges_df = pd.read_csv(edges_csv_path)

    src_list = []
    dst_list = []

    for _, row in edges_df.iterrows():
        src_tx = row["txId1"]
        dst_tx = row["txId2"]

        if src_tx in txid_to_idx and dst_tx in txid_to_idx:
            src_list.append(txid_to_idx[src_tx])
            dst_list.append(txid_to_idx[dst_tx])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


def build_masks(labels, test_size=0.2, random_state=42):
    labeled_idx = []
    labeled_y = []

    for idx, label in enumerate(labels):
        if label != -1:
            labeled_idx.append(idx)
            labeled_y.append(label)

    train_idx, test_idx = train_test_split(
        labeled_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=labeled_y
    )

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    for idx in train_idx:
        train_mask[idx] = True

    for idx in test_idx:
        test_mask[idx] = True

    return train_mask, test_mask


def standardize_tensor(x, train_mask):
    x_train = x[train_mask]

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)

    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    x_standardized = (x - mean) / std

    return x_standardized, mean, std


def build_gnn_data(features_csv_path, edges_csv_path):
    df = pd.read_csv(features_csv_path)

    node_feature_columns = get_node_feature_columns(df)

    for col in RELAY_FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing relay feature column: {col}")

    tx_ids = df["txId"].tolist()
    txid_to_idx = {}

    for idx, tx_id in enumerate(tx_ids):
        txid_to_idx[tx_id] = idx

    y = torch.tensor(df["label"].values, dtype=torch.long)
    train_mask, test_mask = build_masks(df["label"].tolist())

    x_raw = torch.tensor(df[node_feature_columns].values, dtype=torch.float)
    relay_x_raw = torch.tensor(df[RELAY_FEATURE_COLUMNS].values, dtype=torch.float)

    x, x_mean, x_std = standardize_tensor(x_raw, train_mask)
    relay_x, relay_mean, relay_std = standardize_tensor(relay_x_raw, train_mask)

    edge_index = build_edge_index(edges_csv_path, txid_to_idx)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.relay_x = relay_x

    data.tx_ids = tx_ids
    data.node_feature_columns = node_feature_columns
    data.relay_feature_columns = RELAY_FEATURE_COLUMNS

    data.x_mean = x_mean
    data.x_std = x_std
    data.relay_mean = relay_mean
    data.relay_std = relay_std

    return data


if __name__ == "__main__":
    data = build_gnn_data(
        features_csv_path="data/graph_features.csv",
        edges_csv_path="data/elliptic_txs_edgelist.csv"
    )

    print("========== GNN DATA ==========")
    print(data)
    print()

    print("Num nodes:", data.num_nodes)
    print("Num edges:", data.num_edges)
    print("Num node features:", data.x.shape[1])
    print("Num relay features:", data.relay_x.shape[1])
    print()

    print("Train nodes:", int(data.train_mask.sum()))
    print("Test nodes:", int(data.test_mask.sum()))
    print()

    labeled_mask = data.y != -1
    print("Labeled nodes:", int(labeled_mask.sum()))
    print("Illicit labeled nodes:", int((data.y[labeled_mask] == 1).sum()))
    print("Licit labeled nodes:", int((data.y[labeled_mask] == 0).sum()))
