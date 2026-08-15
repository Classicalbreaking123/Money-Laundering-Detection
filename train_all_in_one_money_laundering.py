from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
import heapq
import math
import random
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


try:
    import shap
except ImportError:
    shap = None


SEED = 42
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.15

COMMUNITY_GAMMA = 1.0
COMMUNITY_MAX_PASSES = 15
LAMBDA_COMMUNITY = 0.2

COMMUNITY_HITS_MAX_ITER = 50
COMMUNITY_HITS_TOL = 1e-8


EPOCHS = 600
BASE_OOF_EPOCHS = 300
OOF_FOLDS = 5
LR = 1e-3

DFER_BETA = 0.8
DFER_ALPHA = 0.5
DFER_GAMMA = 0.1
DFER_MAX_ITER = 30
DFER_TOL = 1e-8

ARTIFACT_DIR = Path("artifacts_all_in_one")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES_PATH = "data/elliptic_txs_classes.csv"
EDGELIST_PATH = "data/elliptic_txs_edgelist.csv"
FEATURES_PATH = "data/elliptic_txs_features.csv"

FINAL_SCALER_PATH = ARTIFACT_DIR / "final_all_feature_scaler.pkl"
FINAL_FEATURE_COLS_PATH = ARTIFACT_DIR / "final_feature_columns.pkl"
BASE_FEATURE_COLS_PATH = ARTIFACT_DIR / "base_feature_columns.pkl"

FINAL_FEATURES_OUTPUT_PATH = ARTIFACT_DIR / "all_features_with_custom.csv"
FINAL_SCORES_PATH = ARTIFACT_DIR / "final_all_risk_scores.csv"
BASE_RISK_OUTPUT_PATH = ARTIFACT_DIR / "base_risk_scores.csv"
SHAP_IMPORTANCE_PATH = ARTIFACT_DIR / "shap_feature_importance.csv"
SHAP_VALUES_PATH = ARTIFACT_DIR / "shap_test_values.csv"


@dataclass
class EllipticRawData:
    graph: nx.DiGraph
    node_ids: list
    labels: dict
    timesteps: dict
    in_neighbors: dict
    out_neighbors: dict


def load_elliptic_raw(classes_path, edgelist_path, features_path):
    classes_df = pd.read_csv(classes_path)
    edges_df = pd.read_csv(edgelist_path)
    features_df = pd.read_csv(features_path, header=None)

    if "txId1" in edges_df.columns and "txId2" in edges_df.columns:
        edges_df = edges_df.rename(columns={"txId1": "src", "txId2": "dst"})
    elif "src" not in edges_df.columns or "dst" not in edges_df.columns:
        edges_df = pd.read_csv(edgelist_path, header=None, names=["src", "dst"])

    features_df = features_df.rename(columns={0: "txId", 1: "timestep"})

    classes_df["txId"] = pd.to_numeric(classes_df["txId"], errors="coerce")
    features_df["txId"] = pd.to_numeric(features_df["txId"], errors="coerce")
    edges_df["src"] = pd.to_numeric(edges_df["src"], errors="coerce")
    edges_df["dst"] = pd.to_numeric(edges_df["dst"], errors="coerce")

    classes_df = classes_df.dropna(subset=["txId"]).copy()
    features_df = features_df.dropna(subset=["txId"]).copy()
    edges_df = edges_df.dropna(subset=["src", "dst"]).copy()

    classes_df["txId"] = classes_df["txId"].astype(int)
    features_df["txId"] = features_df["txId"].astype(int)
    edges_df["src"] = edges_df["src"].astype(int)
    edges_df["dst"] = edges_df["dst"].astype(int)

    label_map = {}
    for _, row in classes_df.iterrows():
        label_map[int(row["txId"])] = str(row["class"]).strip()

    timestep_map = {}
    for _, row in features_df.iterrows():
        timestep_map[int(row["txId"])] = int(row["timestep"])

    G = nx.DiGraph()
    all_nodes = set(features_df["txId"].tolist())
    G.add_nodes_from(all_nodes)

    for _, row in edges_df.iterrows():
        G.add_edge(int(row["src"]), int(row["dst"]))

    node_ids = list(features_df["txId"].tolist())

    labels = {}
    timesteps = {}
    in_neighbors = {}
    out_neighbors = {}

    for node in node_ids:
        labels[node] = label_map.get(node, "unknown")
        timesteps[node] = timestep_map[node]
        in_neighbors[node] = list(G.predecessors(node))
        out_neighbors[node] = list(G.successors(node))

    return EllipticRawData(
        graph=G,
        node_ids=node_ids,
        labels=labels,
        timesteps=timesteps,
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
    )


def normalize_label_to_binary(label_value):
    s = str(label_value).strip().lower()

    if s in {"unknown", "nan", "none", ""}:
        return np.nan

    try:
        val = float(s)
    except Exception:
        return np.nan

    if val == 1:
        return 1
    if val == 2:
        return 0
    if val == 0:
        return 0

    return np.nan


def compute_core_numbers_manual(G_und):
    adjacency = {}
    degree = {}

    for node in G_und.nodes():
        nbrs = set(G_und.neighbors(node))
        adjacency[node] = nbrs
        degree[node] = len(nbrs)

    heap = []
    for node in G_und.nodes():
        heapq.heappush(heap, (degree[node], node))

    removed = set()
    core_number = {}
    current_k = 0

    while len(heap) > 0:
        deg, node = heapq.heappop(heap)

        if node in removed:
            continue

        if deg != degree[node]:
            continue

        if deg > current_k:
            current_k = deg

        core_number[node] = current_k
        removed.add(node)

        for nbr in adjacency[node]:
            if nbr not in removed:
                degree[nbr] = degree[nbr] - 1
                heapq.heappush(heap, (degree[nbr], nbr))

    return core_number


def compute_bridge_features(raw_data, G_und, node_ids):
    eps = 1e-9

    und_neighbors = {}
    for node in node_ids:
        und_neighbors[node] = set(G_und.neighbors(node))

    bridge_relay_score = {}
    bridge_cross_density = {}
    bridge_score = {}

    for node in node_ids:
        in_side = set(raw_data.in_neighbors[node])
        out_side = set(raw_data.out_neighbors[node])

        if len(in_side) == 0 or len(out_side) == 0:
            bridge_relay_score[node] = 0.0
            bridge_cross_density[node] = 1.0
            bridge_score[node] = 0.0
            continue

        A = set(in_side)
        for u in in_side:
            A.update(und_neighbors.get(u, set()))
        if node in A:
            A.remove(node)

        B = set(out_side)
        for u in out_side:
            B.update(und_neighbors.get(u, set()))
        if node in B:
            B.remove(node)

        overlap = A & B
        if len(overlap) > 0:
            A = A - overlap
            B = B - overlap

        if len(A) == 0 or len(B) == 0:
            bridge_relay_score[node] = 0.0
            bridge_cross_density[node] = 1.0
            bridge_score[node] = 0.0
            continue

        cross_edges = 0
        for a in A:
            for nbr in und_neighbors.get(a, set()):
                if nbr == node:
                    continue
                if nbr in B:
                    cross_edges = cross_edges + 1

        cross_density = cross_edges / (len(A) * len(B) + eps)
        relay_score = min(len(in_side), len(out_side)) / (max(len(in_side), len(out_side)) + eps)
        final_bridge_score = relay_score * (1.0 - cross_density)

        bridge_relay_score[node] = relay_score
        bridge_cross_density[node] = cross_density
        bridge_score[node] = final_bridge_score

    return bridge_relay_score, bridge_cross_density, bridge_score


def compute_temporal_burst_entropy_score(raw_data, node_ids):
    temporal_burst_entropy_score = {}

    for node in node_ids:
        t_i = raw_data.timesteps[node]
        neighbors = set(raw_data.in_neighbors[node]) | set(raw_data.out_neighbors[node])

        if len(neighbors) == 0:
            temporal_burst_entropy_score[node] = 0.0
            continue

        offsets = []
        for nbr in neighbors:
            offsets.append(raw_data.timesteps[nbr] - t_i)

        counter = Counter(offsets)
        total = float(len(offsets))
        K = len(counter)

        if K <= 1:
            temporal_burst_entropy_score[node] = 1.0
            continue

        entropy = 0.0
        for count in counter.values():
            p = count / total
            entropy = entropy - p * math.log(p + 1e-12)

        max_entropy = math.log(K + 1e-12)
        score = 1.0 - (entropy / (max_entropy + 1e-12))

        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        temporal_burst_entropy_score[node] = score

    return temporal_burst_entropy_score


def compute_discounted_flow_entropy_rate(
    raw_data,
    beta=0.8,
    alpha=0.5,
    gamma=0.1,
    max_iter=30,
    tol=1e-8,
):
    node_ids = raw_data.node_ids

    outdegree = {}
    two_hop_reach = {}

    for node in node_ids:
        out_neighbors = raw_data.out_neighbors[node]
        outdegree[node] = len(out_neighbors)

    for node in node_ids:
        first_hop = raw_data.out_neighbors[node]
        two_hop_nodes = set()

        for nbr in first_hop:
            for nbr2 in raw_data.out_neighbors[nbr]:
                if nbr2 != node:
                    two_hop_nodes.add(nbr2)

        two_hop_reach[node] = len(two_hop_nodes)

    transition_probs = {}
    one_step_entropy = {}

    for node in node_ids:
        children = raw_data.out_neighbors[node]

        if len(children) == 0:
            transition_probs[node] = []
            one_step_entropy[node] = 0.0
            continue

        scores = []
        total_score = 0.0

        for child in children:
            score = 1.0 + alpha * outdegree[child] + gamma * two_hop_reach[child]
            scores.append((child, score))
            total_score = total_score + score

        probs = []
        entropy = 0.0

        for child, score in scores:
            p = score / (total_score + 1e-12)
            probs.append((child, p))
            entropy = entropy - p * math.log(p + 1e-12)

        transition_probs[node] = probs
        one_step_entropy[node] = entropy

    flow_uncertainty = {}
    for node in node_ids:
        flow_uncertainty[node] = one_step_entropy[node]

    for iteration in range(max_iter):
        new_flow_uncertainty = {}
        max_change = 0.0

        for node in node_ids:
            value = one_step_entropy[node]

            expected_future = 0.0
            for child, p in transition_probs[node]:
                expected_future = expected_future + p * flow_uncertainty[child]

            value = value + beta * expected_future
            new_flow_uncertainty[node] = value

            diff = abs(value - flow_uncertainty[node])
            if diff > max_change:
                max_change = diff

        flow_uncertainty = new_flow_uncertainty

        print(
            f"DFER iteration {iteration + 1:02d}/{max_iter} | "
            f"max change = {max_change:.10f}"
        )

        if max_change < tol:
            print("DFER converged early.")
            break

    return flow_uncertainty



def compute_additional_flow_features(raw_data, node_ids, partition):
    """Compute additional fan-in, fan-out, relay, neighbor-structure,
    neighbor-core, neighbor-clustering and community flow features."""
    fan_in_score = {}
    fan_out_score = {}
    relay_score = {}
    in_neighbor_mean_degree = {}
    out_neighbor_mean_degree = {}
    in_neighbor_max_degree = {}
    out_neighbor_max_degree = {}
    in_neighbor_mean_core = {}
    out_neighbor_mean_core = {}
    in_neighbor_max_core = {}
    out_neighbor_max_core = {}
    in_neighbor_mean_clustering = {}
    out_neighbor_mean_clustering = {}
    community_internal_ratio = {}
    community_external_ratio = {}

    G = raw_data.graph
    G_und = G.to_undirected()
    und_degree = dict(G_und.degree())
    neighbor_core = compute_core_numbers_manual(G_und)
    neighbor_clustering = nx.clustering(G_und)

    for node in node_ids:
        in_neighbors = set(raw_data.in_neighbors[node])
        out_neighbors = set(raw_data.out_neighbors[node])

        in_deg = len(in_neighbors)
        out_deg = len(out_neighbors)

        fan_in_score[node] = in_deg / (in_deg + out_deg + 1e-9)
        fan_out_score[node] = out_deg / (in_deg + out_deg + 1e-9)

        relay_score[node] = (
            min(in_deg, out_deg) / (max(in_deg, out_deg) + 1e-9)
        )

        in_degrees = [und_degree.get(nbr, 0) for nbr in in_neighbors]
        out_degrees = [und_degree.get(nbr, 0) for nbr in out_neighbors]

        in_neighbor_mean_degree[node] = (
            float(np.mean(in_degrees)) if in_degrees else 0.0
        )
        out_neighbor_mean_degree[node] = (
            float(np.mean(out_degrees)) if out_degrees else 0.0
        )
        in_neighbor_max_degree[node] = (
            float(max(in_degrees)) if in_degrees else 0.0
        )
        out_neighbor_max_degree[node] = (
            float(max(out_degrees)) if out_degrees else 0.0
        )

        in_cores = [neighbor_core.get(nbr, 0) for nbr in in_neighbors]
        out_cores = [neighbor_core.get(nbr, 0) for nbr in out_neighbors]
        in_cluster = [neighbor_clustering.get(nbr, 0.0) for nbr in in_neighbors]
        out_cluster = [neighbor_clustering.get(nbr, 0.0) for nbr in out_neighbors]

        in_neighbor_mean_core[node] = (
            float(np.mean(in_cores)) if in_cores else 0.0
        )
        out_neighbor_mean_core[node] = (
            float(np.mean(out_cores)) if out_cores else 0.0
        )
        in_neighbor_max_core[node] = (
            float(max(in_cores)) if in_cores else 0.0
        )
        out_neighbor_max_core[node] = (
            float(max(out_cores)) if out_cores else 0.0
        )
        in_neighbor_mean_clustering[node] = (
            float(np.mean(in_cluster)) if in_cluster else 0.0
        )
        out_neighbor_mean_clustering[node] = (
            float(np.mean(out_cluster)) if out_cluster else 0.0
        )

        cid = partition[node]
        total = in_deg + out_deg
        internal = 0

        for nbr in in_neighbors:
            if partition.get(nbr) == cid:
                internal += 1

        for nbr in out_neighbors:
            if partition.get(nbr) == cid:
                internal += 1

        community_internal_ratio[node] = internal / (total + 1e-9)
        community_external_ratio[node] = 1.0 - community_internal_ratio[node]

    return (
        fan_in_score,
        fan_out_score,
        relay_score,
        in_neighbor_mean_degree,
        out_neighbor_mean_degree,
        in_neighbor_max_degree,
        out_neighbor_max_degree,
        in_neighbor_mean_core,
        out_neighbor_mean_core,
        in_neighbor_max_core,
        out_neighbor_max_core,
        in_neighbor_mean_clustering,
        out_neighbor_mean_clustering,
        community_internal_ratio,
        community_external_ratio,
    )


def compute_graph_features(raw_data, partition=None):
    G = raw_data.graph
    node_ids = raw_data.node_ids
    timesteps = raw_data.timesteps

    if partition is None:
        adjacency_for_partition = build_undirected_adjacency(raw_data)
        partition = run_louvain_local_communities(
            adjacency=adjacency_for_partition,
            max_passes=COMMUNITY_MAX_PASSES,
            gamma=COMMUNITY_GAMMA,
        )

    indegree = dict(G.in_degree())
    outdegree = dict(G.out_degree())

    scc_size = {}
    sccs = list(nx.strongly_connected_components(G))
    for comp in sccs:
        size = len(comp)
        for node in comp:
            scc_size[node] = size

    G_und = G.to_undirected()

    core_number = compute_core_numbers_manual(G_und)
    clustering_coeff = nx.clustering(G_und)

    bridge_relay_score, bridge_cross_density, bridge_score = compute_bridge_features(
        raw_data, G_und, node_ids
    )

    temporal_in_degree = {}
    temporal_out_degree = {}

    for node in node_ids:
        in_neighbors = set(raw_data.in_neighbors[node])
        out_neighbors_node = set(raw_data.out_neighbors[node])
        t = timesteps[node]

        temp_in = 0
        for nbr in in_neighbors:
            if abs(timesteps[nbr] - t) <= 1:
                temp_in = temp_in + 1
        temporal_in_degree[node] = temp_in

        temp_out = 0
        for nbr in out_neighbors_node:
            if abs(timesteps[nbr] - t) <= 1:
                temp_out = temp_out + 1
        temporal_out_degree[node] = temp_out

    print("Computing temporal burst entropy feature...")
    temporal_burst_entropy_map = compute_temporal_burst_entropy_score(raw_data, node_ids)

    print("Computing DFER flow uncertainty feature...")
    flow_uncertainty_map = compute_discounted_flow_entropy_rate(
        raw_data,
        beta=DFER_BETA,
        alpha=DFER_ALPHA,
        gamma=DFER_GAMMA,
        max_iter=DFER_MAX_ITER,
        tol=DFER_TOL,
    )

    (
        fan_in_score,
        fan_out_score,
        relay_score,
        in_neighbor_mean_degree,
        out_neighbor_mean_degree,
        in_neighbor_max_degree,
        out_neighbor_max_degree,
        in_neighbor_mean_core,
        out_neighbor_mean_core,
        in_neighbor_max_core,
        out_neighbor_max_core,
        in_neighbor_mean_clustering,
        out_neighbor_mean_clustering,
        community_internal_ratio,
        community_external_ratio,
    ) = compute_additional_flow_features(
        raw_data, node_ids, partition
    )

    rows = []

    for node in node_ids:
        in_deg = indegree.get(node, 0)
        out_deg = outdegree.get(node, 0)

        degree_imbalance = (out_deg - in_deg) / (out_deg + in_deg + 1e-9)
        pass_through = min(in_deg, out_deg) / (max(in_deg, out_deg) + 1e-9)
        split_score = out_deg / (in_deg + 1.0)

        temp_in = temporal_in_degree[node]
        temp_out = temporal_out_degree[node]
        temporal_pass_through = min(temp_in, temp_out) / (max(temp_in, temp_out) + 1e-9)

        row = {
            "txId": node,
            "timestep": raw_data.timesteps[node],
            "label": raw_data.labels[node],
            "label_binary": normalize_label_to_binary(raw_data.labels[node]),
            "indegree": in_deg,
            "outdegree": out_deg,
            "degree_imbalance": degree_imbalance,
            "pass_through": pass_through,
            "split_score": split_score,
            "scc_size": scc_size.get(node, 1),
            "core_number": core_number.get(node, 0),
            "temporal_burst_entropy_score": temporal_burst_entropy_map[node],
            "temporal_in_degree": temp_in,
            "temporal_out_degree": temp_out,
            "temporal_pass_through": temporal_pass_through,
            "clustering_coeff": clustering_coeff.get(node, 0.0),
            "bridge_relay_score": bridge_relay_score.get(node, 0.0),
            "bridge_cross_density": bridge_cross_density.get(node, 1.0),
            "bridge_score": bridge_score.get(node, 0.0),
            "flow_uncertainty_score": flow_uncertainty_map[node],
            "fan_in_score": fan_in_score[node],
            "fan_out_score": fan_out_score[node],
            "relay_score": relay_score[node],
            "in_neighbor_mean_degree": in_neighbor_mean_degree[node],
            "out_neighbor_mean_degree": out_neighbor_mean_degree[node],
            "in_neighbor_max_degree": in_neighbor_max_degree[node],
            "out_neighbor_max_degree": out_neighbor_max_degree[node],
            "in_neighbor_mean_core": in_neighbor_mean_core[node],
            "out_neighbor_mean_core": out_neighbor_mean_core[node],
            "in_neighbor_max_core": in_neighbor_max_core[node],
            "out_neighbor_max_core": out_neighbor_max_core[node],
            "in_neighbor_mean_clustering": in_neighbor_mean_clustering[node],
            "out_neighbor_mean_clustering": out_neighbor_mean_clustering[node],
            "community_internal_ratio": community_internal_ratio[node],
            "community_external_ratio": community_external_ratio[node],
        }
        rows.append(row)

    return pd.DataFrame(rows)


class RiskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    epochs=60,
    lr=1e-3,
    hidden_dims=(128, 64),
    dropout=0.2,
):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32)

    model = RiskMLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    num_pos = float(y_train.sum())
    num_neg = float(len(y_train) - y_train.sum())
    pos_weight_value = num_neg / (num_pos + 1e-9)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_validation_loss = float("inf")
    best_epoch = -1
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        train_logits = model(X_train_tensor)
        train_loss = criterion(train_logits, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(X_validation_tensor)
            validation_loss = criterion(
                validation_logits,
                y_validation_tensor,
            )

        validation_loss_value = validation_loss.item()

        if validation_loss_value < best_validation_loss:
            best_validation_loss = validation_loss_value
            best_epoch = epoch + 1
            best_model_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Validation Loss: {validation_loss_value:.4f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_epoch, best_validation_loss

def build_undirected_adjacency(raw_data):
    adjacency = {}
    for node in raw_data.node_ids:
        adjacency[node] = set()

    for node in raw_data.node_ids:
        for nbr in raw_data.out_neighbors[node]:
            adjacency[node].add(nbr)
            adjacency[nbr].add(node)

    return adjacency


def compute_degrees(adjacency):
    degree = {}
    for node, nbrs in adjacency.items():
        degree[node] = len(nbrs)
    return degree


def initialize_singleton_partition(node_ids):
    partition = {}
    for idx, node in enumerate(node_ids):
        partition[node] = idx
    return partition


def relabel_partition(partition):
    unique_ids = sorted(set(partition.values()))
    mapping = {}
    for i, cid in enumerate(unique_ids):
        mapping[cid] = i

    new_partition = {}
    for node, cid in partition.items():
        new_partition[node] = mapping[cid]
    return new_partition


def compute_community_degree_sums(adjacency, partition, degree):
    comm_degree_sum = defaultdict(float)
    for node, cid in partition.items():
        comm_degree_sum[cid] += degree[node]
    return comm_degree_sum


def compute_neighbor_comm_edge_counts(node, adjacency, partition):
    counts = defaultdict(int)
    for nbr in adjacency[node]:
        nbr_cid = partition[nbr]
        counts[nbr_cid] += 1
    return counts


def run_louvain_local_communities(adjacency, max_passes=15, gamma=1.0):
    node_ids = list(adjacency.keys())
    partition = initialize_singleton_partition(node_ids)
    partition = relabel_partition(partition)

    degree = compute_degrees(adjacency)
    m = sum(len(adjacency[node]) for node in adjacency) / 2.0

    if m == 0:
        return partition

    for _ in range(max_passes):
        moved_any = False
        random.shuffle(node_ids)

        comm_degree_sum = compute_community_degree_sums(adjacency, partition, degree)

        for node in node_ids:
            current_cid = partition[node]
            k_i = degree[node]

            comm_degree_sum[current_cid] -= k_i

            neighbor_comms = set()
            for nbr in adjacency[node]:
                neighbor_comms.add(partition[nbr])

            best_cid = current_cid
            best_gain = 0.0

            nbr_comm_counts = compute_neighbor_comm_edge_counts(node, adjacency, partition)

            for target_cid in neighbor_comms:
                if target_cid == current_cid:
                    continue

                k_i_in_target = nbr_comm_counts.get(target_cid, 0)
                sigma_target = comm_degree_sum[target_cid]

                gain = k_i_in_target - gamma * (k_i * sigma_target) / (2.0 * m)

                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_cid = target_cid

            if best_cid != current_cid:
                partition[node] = best_cid
                moved_any = True

            new_cid = partition[node]
            comm_degree_sum[new_cid] += k_i

        partition = relabel_partition(partition)

        if not moved_any:
            break

    partition = relabel_partition(partition)
    return partition


def get_community_nodes(partition):
    comm_to_nodes = defaultdict(list)
    for node, cid in partition.items():
        comm_to_nodes[cid].append(node)
    return comm_to_nodes


def minmax_scale_dict(values_dict, eps=1e-12):
    vals = list(values_dict.values())
    if len(vals) == 0:
        return {}

    mn = min(vals)
    mx = max(vals)

    scaled = {}
    if mx - mn <= eps:
        for key in values_dict:
            scaled[key] = 0.0
        return scaled

    for key, val in values_dict.items():
        scaled[key] = (val - mn) / (mx - mn + eps)

    return scaled


def compute_hits_for_community(
    community_nodes,
    raw_data,
    max_iter=50,
    tol=1e-8,
):
    """Standard HITS computed independently of any ML risk score."""
    node_set = set(community_nodes)

    in_neighbors_comm = {
        node: [nbr for nbr in raw_data.in_neighbors[node] if nbr in node_set]
        for node in community_nodes
    }
    out_neighbors_comm = {
        node: [nbr for nbr in raw_data.out_neighbors[node] if nbr in node_set]
        for node in community_nodes
    }

    hub = {node: 1.0 for node in community_nodes}
    authority = {node: 1.0 for node in community_nodes}

    for _ in range(max_iter):
        new_authority = {}
        for node in community_nodes:
            new_authority[node] = sum(hub[nbr] for nbr in in_neighbors_comm[node])

        auth_norm = math.sqrt(sum(v * v for v in new_authority.values()))
        if auth_norm > 0:
            for node in community_nodes:
                new_authority[node] /= auth_norm

        new_hub = {}
        for node in community_nodes:
            new_hub[node] = sum(
                new_authority[nbr] for nbr in out_neighbors_comm[node]
            )

        hub_norm = math.sqrt(sum(v * v for v in new_hub.values()))
        if hub_norm > 0:
            for node in community_nodes:
                new_hub[node] /= hub_norm

        diff = 0.0
        for node in community_nodes:
            diff += abs(new_authority[node] - authority[node])
            diff += abs(new_hub[node] - hub[node])

        authority = new_authority
        hub = new_hub

        if diff < tol:
            break

    return hub, authority


def compute_community_hits_features(
    partition,
    raw_data,
    max_iter=50,
    tol=1e-8,
):
    """Create community size, hub, authority and combined HITS features."""
    comm_to_nodes = get_community_nodes(partition)

    community_size_by_cid = {}
    community_hub_score = {}
    community_authority_score = {}
    community_hits_score = {}

    for cid, nodes in comm_to_nodes.items():
        community_size_by_cid[cid] = len(nodes)

        if len(nodes) == 1:
            node = nodes[0]
            community_hub_score[node] = 0.0
            community_authority_score[node] = 0.0
            community_hits_score[node] = 0.0
            continue

        hub_dict, auth_dict = compute_hits_for_community(
            nodes, raw_data, max_iter=max_iter, tol=tol
        )

        combined = {
            node: 0.5 * (hub_dict[node] + auth_dict[node])
            for node in nodes
        }
        normalized = minmax_scale_dict(combined)

        for node in nodes:
            community_hub_score[node] = hub_dict[node]
            community_authority_score[node] = auth_dict[node]
            community_hits_score[node] = normalized[node]

    return (
        community_size_by_cid,
        community_hub_score,
        community_authority_score,
        community_hits_score,
    )


def compute_risk_aware_hits_for_community(
    community_nodes,
    raw_data,
    base_risk_map,
    max_iter=50,
    tol=1e-8,
):
   
    node_set = set(community_nodes)

    in_neighbors_comm = {
        node: [nbr for nbr in raw_data.in_neighbors[node] if nbr in node_set]
        for node in community_nodes
    }
    out_neighbors_comm = {
        node: [nbr for nbr in raw_data.out_neighbors[node] if nbr in node_set]
        for node in community_nodes
    }

    hub = {node: 1.0 for node in community_nodes}
    authority = {node: 1.0 for node in community_nodes}

    for _ in range(max_iter):
        new_authority = {}
        for node in community_nodes:
            value = 0.0
            for nbr in in_neighbors_comm[node]:
                value += base_risk_map[nbr] * hub[nbr]
            new_authority[node] = value

        auth_norm = math.sqrt(sum(v * v for v in new_authority.values()))
        if auth_norm > 0:
            for node in community_nodes:
                new_authority[node] /= auth_norm

        new_hub = {}
        for node in community_nodes:
            value = 0.0
            for nbr in out_neighbors_comm[node]:
                value += base_risk_map[nbr] * new_authority[nbr]
            new_hub[node] = value

        hub_norm = math.sqrt(sum(v * v for v in new_hub.values()))
        if hub_norm > 0:
            for node in community_nodes:
                new_hub[node] /= hub_norm

        diff = 0.0
        for node in community_nodes:
            diff += abs(new_authority[node] - authority[node])
            diff += abs(new_hub[node] - hub[node])

        authority = new_authority
        hub = new_hub

        if diff < tol:
            break

    return hub, authority


def compute_community_hits_refinement(
    partition,
    raw_data,
    base_risk_map,
    lam=0.1,
    max_iter=50,
    tol=1e-8,
):
    comm_to_nodes = get_community_nodes(partition)

    community_size_by_cid = {}
    community_hub_score = {}
    community_authority_score = {}
    community_hits_score = {}
    community_refined_risk = {}

    for cid, nodes in comm_to_nodes.items():
        community_size_by_cid[cid] = len(nodes)

        if len(nodes) == 1:
            node = nodes[0]
            community_hub_score[node] = 0.0
            community_authority_score[node] = 0.0
            community_hits_score[node] = 0.0
            community_refined_risk[node] = base_risk_map[node]
            continue

        hub_dict, auth_dict = compute_risk_aware_hits_for_community(
            community_nodes=nodes,
            raw_data=raw_data,
            base_risk_map=base_risk_map,
            max_iter=max_iter,
            tol=tol,
        )

        combined_score = {}
        for node in nodes:
            community_hub_score[node] = hub_dict[node]
            community_authority_score[node] = auth_dict[node]
            combined_score[node] = 0.5 * (hub_dict[node] + auth_dict[node])

        normalized_combined = minmax_scale_dict(combined_score)

        for node in nodes:
            c_i = normalized_combined[node]
            b_i = base_risk_map[node]

            community_hits_score[node] = c_i
            community_refined_risk[node] = (b_i + lam * c_i) / (1.0 + lam)

    return (
        community_size_by_cid,
        community_hub_score,
        community_authority_score,
        community_hits_score,
        community_refined_risk,
    )


def compute_shap_feature_importance(model, X_background, X_explain, feature_names):

    if shap is None:
        raise ImportError(
            "The 'shap' package is not installed. Install it with: pip install shap"
        )

    model.eval()

    background = torch.tensor(X_background, dtype=torch.float32)
    explain_tensor = torch.tensor(X_explain, dtype=torch.float32)

    class ShapModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            return self.base_model(x).unsqueeze(1)

    shap_model = ShapModelWrapper(model)

    explainer = shap.DeepExplainer(shap_model, background)
    shap_values = explainer.shap_values(explain_tensor)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]
    elif shap_values.ndim == 3 and shap_values.shape[0] == 1:
        shap_values = shap_values[0]

    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP feature dimension {shap_values.shape[1]} does not match "
            f"{len(feature_names)} feature names."
        )

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    return importance_df, shap_values


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    raw_data = load_elliptic_raw(
        classes_path=CLASSES_PATH,
        edgelist_path=EDGELIST_PATH,
        features_path=FEATURES_PATH,
    )

    print("Building graph features")
    features_df = compute_graph_features(raw_data)

    labeled_df = features_df[features_df["label_binary"].notna()].copy()
    labeled_df["target"] = labeled_df["label_binary"].astype(int)

    print("Total rows      :", len(features_df))
    print("Labeled rows    :", len(labeled_df))
    print("Illicit labeled :", int((labeled_df["target"] == 1).sum()))
    print("Licit labeled   :", int((labeled_df["target"] == 0).sum()))

    train_df, test_df = train_test_split(
        labeled_df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labeled_df["target"],
    )

    train_df, validation_df = train_test_split(
        train_df,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=train_df["target"],
    )

    train_txids = set(train_df["txId"].astype(int))
    validation_txids = set(validation_df["txId"].astype(int))
    test_txids = set(test_df["txId"].astype(int))

    print("Running community detection...")

    adjacency = build_undirected_adjacency(raw_data)

    partition = run_louvain_local_communities(
        adjacency=adjacency,
        max_passes=COMMUNITY_MAX_PASSES,
        gamma=COMMUNITY_GAMMA,
    )

    features_df = compute_graph_features(raw_data, partition=partition)


    print("\nBUILDING LEAKAGE-SAFE BASE RISK SCORES")

    HITS_FEATURE_NAMES = {
        "community_hub_score",
        "community_authority_score",
        "community_hits_score",
        "community_refined_risk",
        "base_risk_score",
    }

    base_feature_columns = [
        col for col in features_df.columns
        if col not in {
            "txId",
            "label",
            "label_binary",
            "target",
            "community_id",
        } and col not in HITS_FEATURE_NAMES
    ]

    with open(BASE_FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(base_feature_columns, f)

    print("Number of base-model features:", len(base_feature_columns))

    def get_scaled_matrix(df, columns, scaler):
        X = df[columns].fillna(0.0).values.astype(np.float32)
        return scaler.transform(X).astype(np.float32)

    def train_base_and_score(
        base_train_df,
        base_validation_df,
        score_df,
        epochs,
        run_name,
    ):
        X_base_train = base_train_df[base_feature_columns].fillna(0.0).values.astype(np.float32)
        y_base_train = base_train_df["target"].values.astype(np.float32)

        X_base_val = base_validation_df[base_feature_columns].fillna(0.0).values.astype(np.float32)
        y_base_val = base_validation_df["target"].values.astype(np.float32)

        scaler = StandardScaler()
        X_base_train = scaler.fit_transform(X_base_train).astype(np.float32)
        X_base_val = scaler.transform(X_base_val).astype(np.float32)

        model, best_epoch, best_loss = train_model(
            X_base_train,
            y_base_train,
            X_base_val,
            y_base_val,
            epochs=epochs,
            lr=LR,
            hidden_dims=(256, 128, 64),
            dropout=0.25,
        )

        X_score = score_df[base_feature_columns].fillna(0.0).values.astype(np.float32)
        X_score = scaler.transform(X_score).astype(np.float32)

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_score, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()

        print(
            f"{run_name}: best epoch={best_epoch}, "
            f"best validation loss={best_loss:.4f}"
        )

        return probs

    all_node_ids = features_df["txId"].astype(int).tolist()
    train_array = train_df.reset_index(drop=True)
    train_y = train_array["target"].astype(int).values

    skf = StratifiedKFold(
        n_splits=OOF_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_risk_maps = {}

    print(
        f"Generating {OOF_FOLDS}-fold out-of-fold base risks "
        "and risk-aware HITS in one pass..."
    )

    train_fold_by_txid = {}
    for fold_id, (_, holdout_idx) in enumerate(
        skf.split(train_array, train_y), start=1
    ):
        for txid in train_array.iloc[holdout_idx]["txId"].astype(int):
            train_fold_by_txid[int(txid)] = fold_id

    for fold_id, (fit_idx, holdout_idx) in enumerate(
        skf.split(train_array, train_y), start=1
    ):
        fold_fit_df = train_array.iloc[fit_idx].copy()
        fold_holdout_df = train_array.iloc[holdout_idx].copy()

        fold_scores = train_base_and_score(
            base_train_df=fold_fit_df,
            base_validation_df=fold_holdout_df,
            score_df=features_df,
            epochs=BASE_OOF_EPOCHS,
            run_name=f"OOF fold {fold_id}/{OOF_FOLDS}",
        )

        fold_risk_maps[fold_id] = dict(zip(all_node_ids, fold_scores))

    fully_oof_risk_map = {}

    for txid in train_df["txId"].astype(int):
        txid = int(txid)
        fold_id = train_fold_by_txid[txid]
        fully_oof_risk_map[txid] = fold_risk_maps[fold_id][txid]

    if len(fully_oof_risk_map) != len(train_df):
        raise RuntimeError(
            "Fully OOF risk map failed: every training node must receive "
            "exactly one out-of-fold base-risk prediction."
        )

    print(
        "\nTraining one full base model for validation/test scoring..."
    )

    full_base_scores = train_base_and_score(
        base_train_df=train_df,
        base_validation_df=validation_df,
        score_df=features_df,
        epochs=EPOCHS,
        run_name="Full base model",
    )

    full_base_score_map = dict(zip(all_node_ids, full_base_scores))

    training_hits_risk_map = dict(full_base_score_map)
    training_hits_risk_map.update(fully_oof_risk_map)

    def compute_risk_hits_for_risk_map(base_risk_map):
        (
            community_size_by_cid,
            community_hub_score,
            community_authority_score,
            community_hits_score,
            community_refined_risk,
        ) = compute_community_hits_refinement(
            partition=partition,
            raw_data=raw_data,
            base_risk_map=base_risk_map,
            lam=LAMBDA_COMMUNITY,
            max_iter=COMMUNITY_HITS_MAX_ITER,
            tol=COMMUNITY_HITS_TOL,
        )

        return (
            community_size_by_cid,
            community_hub_score,
            community_authority_score,
            community_hits_score,
            community_refined_risk,
        )

    (
        train_comm_size,
        train_hub,
        train_auth,
        train_hits,
        train_refined,
    ) = compute_risk_hits_for_risk_map(training_hits_risk_map)

    train_risk_hits = {}

    for txid in train_df["txId"].astype(int):
        txid = int(txid)
        train_risk_hits[txid] = {
            "base_risk_score": fully_oof_risk_map[txid],
            "community_hub_score": train_hub[txid],
            "community_authority_score": train_auth[txid],
            "community_hits_score": train_hits[txid],
            "community_refined_risk": train_refined[txid],
            "community_size": train_comm_size[partition[txid]],
        }

    if len(train_risk_hits) != len(train_df):
        raise RuntimeError(
            "OOF HITS generation failed: not every training node received "
            "exactly one leakage-safe HITS feature set."
        )

    (
        validation_comm_size,
        validation_hub,
        validation_auth,
        validation_hits,
        validation_refined,
    ) = compute_risk_hits_for_risk_map(full_base_score_map)

    validation_risk_hits = {}

    for txid in validation_df["txId"].astype(int):
        txid = int(txid)
        validation_risk_hits[txid] = {
            "base_risk_score": full_base_score_map[txid],
            "community_hub_score": validation_hub[txid],
            "community_authority_score": validation_auth[txid],
            "community_hits_score": validation_hits[txid],
            "community_refined_risk": validation_refined[txid],
            "community_size": validation_comm_size[partition[txid]],
        }

    (
        test_comm_size,
        test_hub,
        test_auth,
        test_hits,
        test_refined,
    ) = compute_risk_hits_for_risk_map(full_base_score_map)

    test_risk_hits = {}

    for txid in test_df["txId"].astype(int):
        txid = int(txid)
        test_risk_hits[txid] = {
            "base_risk_score": full_base_score_map[txid],
            "community_hub_score": test_hub[txid],
            "community_authority_score": test_auth[txid],
            "community_hits_score": test_hits[txid],
            "community_refined_risk": test_refined[txid],
            "community_size": test_comm_size[partition[txid]],
        }

    for feature_name in [
        "base_risk_score",
        "community_hub_score",
        "community_authority_score",
        "community_hits_score",
        "community_refined_risk",
    ]:
        values = []
        for txid in features_df["txId"].astype(int):
            if txid in train_risk_hits:
                values.append(train_risk_hits[txid][feature_name])
            elif txid in validation_risk_hits:
                values.append(validation_risk_hits[txid][feature_name])
            elif txid in test_risk_hits:
                values.append(test_risk_hits[txid][feature_name])
            else:
                values.append(full_base_score_map.get(txid, 0.0))

        features_df[feature_name] = values

    print("\nRISK-AWARE HITS FEATURES ADDED")
    print("base_risk_score")
    print("community_hub_score")
    print("community_authority_score")
    print("community_hits_score")
    print("community_refined_risk")


    risk_by_txid = dict(
        zip(
            features_df["txId"].astype(int),
            features_df["base_risk_score"].astype(float),
        )
    )

    in_neighbor_mean_base_risk = {}
    in_neighbor_max_base_risk = {}
    out_neighbor_mean_base_risk = {}
    risk_weighted_in_degree = {}

    for node in features_df["txId"].astype(int):
        in_neighbors = raw_data.in_neighbors.get(int(node), [])
        out_neighbors = raw_data.out_neighbors.get(int(node), [])

        in_risks = [
            risk_by_txid.get(int(nbr), 0.0)
            for nbr in in_neighbors
        ]
        out_risks = [
            risk_by_txid.get(int(nbr), 0.0)
            for nbr in out_neighbors
        ]

        in_neighbor_mean_base_risk[int(node)] = (
            float(np.mean(in_risks)) if in_risks else 0.0
        )
        in_neighbor_max_base_risk[int(node)] = (
            float(max(in_risks)) if in_risks else 0.0
        )
        out_neighbor_mean_base_risk[int(node)] = (
            float(np.mean(out_risks)) if out_risks else 0.0
        )

        risk_weighted_in_degree[int(node)] = float(sum(in_risks))

    features_df["in_neighbor_mean_base_risk"] = [
        in_neighbor_mean_base_risk[int(txid)]
        for txid in features_df["txId"].astype(int)
    ]
    features_df["in_neighbor_max_base_risk"] = [
        in_neighbor_max_base_risk[int(txid)]
        for txid in features_df["txId"].astype(int)
    ]
    features_df["out_neighbor_mean_base_risk"] = [
        out_neighbor_mean_base_risk[int(txid)]
        for txid in features_df["txId"].astype(int)
    ]
    features_df["risk_weighted_in_degree"] = [
        risk_weighted_in_degree[int(txid)]
        for txid in features_df["txId"].astype(int)
    ]

    print("\nBASE-RISK NEIGHBOR FEATURES ADDED")
    print("in_neighbor_mean_base_risk")
    print("in_neighbor_max_base_risk")
    print("out_neighbor_mean_base_risk")
    print("risk_weighted_in_degree")

    labeled_df = features_df[features_df["label_binary"].notna()].copy()
    labeled_df["target"] = labeled_df["label_binary"].astype(int)

    train_df = labeled_df[
        labeled_df["txId"].astype(int).isin(train_txids)
    ].copy()
    validation_df = labeled_df[
        labeled_df["txId"].astype(int).isin(validation_txids)
    ].copy()
    test_df = labeled_df[
        labeled_df["txId"].astype(int).isin(test_txids)
    ].copy()

    feature_columns = [
        col for col in labeled_df.columns
        if col not in {
            "txId",
            "label",
            "label_binary",
            "target",
            "community_id",
        }
    ]

    print("\nFINAL FEATURE COLUMNS")
    for col in feature_columns:
        print(col)
    print("Number of final features:", len(feature_columns))

    X_train = train_df[feature_columns].fillna(0.0).values.astype(np.float32)
    y_train = train_df["target"].values.astype(np.float32)

    X_validation = (
        validation_df[feature_columns].fillna(0.0).values.astype(np.float32)
    )
    y_validation = validation_df["target"].values.astype(np.float32)

    X_test = test_df[feature_columns].fillna(0.0).values.astype(np.float32)
    y_test = test_df["target"].values.astype(np.float32)

    final_scaler = StandardScaler()
    X_train = final_scaler.fit_transform(X_train).astype(np.float32)
    X_validation = final_scaler.transform(X_validation).astype(np.float32)
    X_test = final_scaler.transform(X_test).astype(np.float32)

    with open(FINAL_SCALER_PATH, "wb") as f:
        pickle.dump(final_scaler, f)

    with open(FINAL_FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(feature_columns, f)

    print("\nTRAINING SINGLE FINAL MODEL")

    final_model, final_best_epoch, final_best_validation_loss = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=EPOCHS,
        lr=LR,
        hidden_dims=(256, 128, 64),
        dropout=0.25,
    )

    print("Best model epoch:", final_best_epoch)
    print(
        "Best model validation loss:",
        f"{final_best_validation_loss:.4f}",
    )

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    final_model.eval()
    with torch.no_grad():
        test_logits = final_model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    test_preds = (test_probs >= 0.7).astype(int)

    cm = confusion_matrix(y_test.astype(int), test_preds)
    report = classification_report(
        y_test.astype(int), test_preds, digits=4
    )
    roc_auc = roc_auc_score(y_test.astype(int), test_probs)

    print("\nFINAL TEST RESULTS")
    print("CONFUSION MATRIX")
    print(cm)
    print("\nCLASSIFICATION REPORT")
    print(report)
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nCOMPUTING SHAP FEATURE IMPORTANCE")

    SHAP_BACKGROUND_SIZE = min(50, len(X_train))
    SHAP_EXPLAIN_SIZE = min(100, len(X_test))

    shap_background = X_train[:SHAP_BACKGROUND_SIZE]

    if len(X_test) > SHAP_EXPLAIN_SIZE:
        shap_indices = np.linspace(
            0, len(X_test) - 1, SHAP_EXPLAIN_SIZE, dtype=int
        )
        shap_explain = X_test[shap_indices]
        shap_txids = test_df.iloc[shap_indices]["txId"].astype(int).values
    else:
        shap_explain = X_test
        shap_txids = test_df["txId"].astype(int).values

    shap_importance_df, shap_values = compute_shap_feature_importance(
        model=final_model,
        X_background=shap_background,
        X_explain=shap_explain,
        feature_names=feature_columns,
    )

    shap_importance_df.to_csv(SHAP_IMPORTANCE_PATH, index=False)

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=feature_columns,
    )
    shap_values_df.insert(0, "txId", shap_txids)
    shap_values_df.to_csv(SHAP_VALUES_PATH, index=False)

    print("\nTOP SHAP FEATURES")
    print(shap_importance_df.head(15).to_string(index=False))

    features_df.to_csv(FINAL_FEATURES_OUTPUT_PATH, index=False)

    X_all = features_df[feature_columns].fillna(0.0).values.astype(np.float32)
    X_all = final_scaler.transform(X_all).astype(np.float32)
    X_all_tensor = torch.tensor(X_all, dtype=torch.float32)

    final_model.eval()
    with torch.no_grad():
        all_logits = final_model(X_all_tensor)
        all_probs = torch.sigmoid(all_logits).cpu().numpy()

    final_scores_df = pd.DataFrame({
        "txId": features_df["txId"].astype(int),
        "final_risk_score": all_probs,
    })
    final_scores_df.to_csv(FINAL_SCORES_PATH, index=False)

    base_risk_output = pd.DataFrame({
        "txId": features_df["txId"].astype(int),
        "base_risk_score": features_df["base_risk_score"].astype(float),
    })
    base_risk_output.to_csv(BASE_RISK_OUTPUT_PATH, index=False)

    print("\nSAVED ARTIFACTS")
    print("Final scaler     :", FINAL_SCALER_PATH)
    print("Final feature cols:", FINAL_FEATURE_COLS_PATH)
    print("All features CSV :", FINAL_FEATURES_OUTPUT_PATH)
    print("Final risk scores :", FINAL_SCORES_PATH)
    print("Base risk scores  :", BASE_RISK_OUTPUT_PATH)
    print("\nDone.")
