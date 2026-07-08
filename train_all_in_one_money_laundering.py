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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


SEED = 42
TEST_SIZE = 0.30

COMMUNITY_GAMMA = 1.0
COMMUNITY_MAX_PASSES = 15
LAMBDA_COMMUNITY = 0.1

COMMUNITY_HITS_MAX_ITER = 50
COMMUNITY_HITS_TOL = 1e-8

BASE_EPOCHS = 300
FINAL_EPOCHS = 450
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

BASE_CHECKPOINT_PATH = ARTIFACT_DIR / "base_checkpoint.pt"
BASE_BEST_MODEL_PATH = ARTIFACT_DIR / "base_best_model.pt"
BASE_SCALER_PATH = ARTIFACT_DIR / "base_scaler.pkl"
BASE_FEATURE_COLS_PATH = ARTIFACT_DIR / "base_feature_columns.pkl"

FINAL_CHECKPOINT_PATH = ARTIFACT_DIR / "final_checkpoint.pt"
FINAL_BEST_MODEL_PATH = ARTIFACT_DIR / "final_best_model.pt"
FINAL_SCALER_PATH = ARTIFACT_DIR / "final_all_feature_scaler.pkl"
FINAL_FEATURE_COLS_PATH = ARTIFACT_DIR / "final_feature_columns.pkl"

FINAL_FEATURES_OUTPUT_PATH = ARTIFACT_DIR / "all_features_with_custom.csv"
FINAL_SCORES_PATH = ARTIFACT_DIR / "final_all_risk_scores.csv"


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


def compute_graph_features(raw_data):
    G = raw_data.graph
    node_ids = raw_data.node_ids
    timesteps = raw_data.timesteps

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


def train_model_with_resume(
    X_train,
    y_train,
    X_test,
    y_test,
    checkpoint_path,
    best_model_path,
    epochs=60,
    lr=1e-3,
    hidden_dims=(128, 64),
    dropout=0.2,
    model_name="model",
):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = RiskMLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)

    num_pos = float(y_train.sum())
    num_neg = float(len(y_train) - y_train.sum())
    pos_weight_value = num_neg / (num_pos + 1e-9)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_test_loss = float("inf")
    best_epoch = -1

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_test_loss = float(checkpoint["best_test_loss"])
        best_epoch = int(checkpoint["best_epoch"])

        print(
            f"Resuming {model_name} from checkpoint: "
            f"epoch {start_epoch}/{epochs} | "
            f"best_test_loss={best_test_loss:.4f} | best_epoch={best_epoch}"
        )
    else:
        print(f"No existing {model_name} checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()

        train_logits = model(X_train_tensor)
        train_loss = criterion(train_logits, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor)
            test_loss = criterion(test_logits, y_test_tensor)

        test_loss_value = test_loss.item()

        if test_loss_value < best_test_loss:
            best_test_loss = test_loss_value
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_test_loss": best_test_loss,
            "best_epoch": best_epoch,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "input_dim": X_train.shape[1],
        }
        torch.save(checkpoint, checkpoint_path)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Test Loss: {test_loss_value:.4f}"
        )

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location="cpu"))

    return model, best_epoch, best_test_loss


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


def compute_risk_aware_hits_for_community(
    community_nodes,
    raw_data,
    base_risk_map,
    max_iter=50,
    tol=1e-8,
):
    node_set = set(community_nodes)

    in_neighbors_comm = {}
    out_neighbors_comm = {}

    for node in community_nodes:
        in_neighbors_comm[node] = [
            nbr for nbr in raw_data.in_neighbors[node]
            if nbr in node_set
        ]
        out_neighbors_comm[node] = [
            nbr for nbr in raw_data.out_neighbors[node]
            if nbr in node_set
        ]

    hub = {}
    authority = {}

    for node in community_nodes:
        hub[node] = 1.0
        authority[node] = 1.0

    for _ in range(max_iter):
        new_authority = {}
        for node in community_nodes:
            score = 0.0
            for nbr in in_neighbors_comm[node]:
                score = score + base_risk_map[nbr] * hub[nbr]
            new_authority[node] = score

        auth_norm = math.sqrt(sum(val * val for val in new_authority.values()))
        if auth_norm > 0:
            for node in community_nodes:
                new_authority[node] = new_authority[node] / auth_norm

        new_hub = {}
        for node in community_nodes:
            score = 0.0
            for nbr in out_neighbors_comm[node]:
                score = score + base_risk_map[nbr] * new_authority[nbr]
            new_hub[node] = score

        hub_norm = math.sqrt(sum(val * val for val in new_hub.values()))
        if hub_norm > 0:
            for node in community_nodes:
                new_hub[node] = new_hub[node] / hub_norm

        diff = 0.0
        for node in community_nodes:
            diff = diff + abs(new_authority[node] - authority[node])
            diff = diff + abs(new_hub[node] - hub[node])

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


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    raw_data = load_elliptic_raw(
        classes_path=CLASSES_PATH,
        edgelist_path=EDGELIST_PATH,
        features_path=FEATURES_PATH
    )

    print("Building graph features...")
    features_df = compute_graph_features(raw_data)

    print("\n========== GRAPH FEATURES BUILT ==========")
    print(features_df.head())
    print()
    print("Shape:", features_df.shape)
    print()

    labeled_df = features_df[features_df["label_binary"].notna()].copy()
    labeled_df["target"] = labeled_df["label_binary"].astype(int)

    print("========== LABEL SUMMARY ==========")
    print("Total rows      :", len(features_df))
    print("Labeled rows    :", len(labeled_df))
    print("Illicit labeled :", int((labeled_df["target"] == 1).sum()))
    print("Licit labeled   :", int((labeled_df["target"] == 0).sum()))
    print()

    train_df, test_df = train_test_split(
        labeled_df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labeled_df["target"]
    )

    train_txids = set(train_df["txId"].astype(int).tolist())
    test_txids = set(test_df["txId"].astype(int).tolist())

    print("========== FIXED SPLIT STATS ==========")
    print("Train size     :", len(train_df))
    print("Test size      :", len(test_df))
    print("Train illicit  :", int((train_df["target"] == 1).sum()))
    print("Train licit    :", int((train_df["target"] == 0).sum()))
    print("Test illicit   :", int((test_df["target"] == 1).sum()))
    print("Test licit     :", int((test_df["target"] == 0).sum()))
    print()

    labeled_df = features_df[features_df["label_binary"].notna()].copy()
    labeled_df["target"] = labeled_df["label_binary"].astype(int)

    train_df = labeled_df[labeled_df["txId"].astype(int).isin(train_txids)].copy()
    test_df = labeled_df[labeled_df["txId"].astype(int).isin(test_txids)].copy()

    base_feature_columns = [
        col for col in labeled_df.columns
        if col not in {
            "txId",
            "label",
            "label_binary",
            "target",
        }
    ]

    X_train_base = train_df[base_feature_columns].fillna(0.0).values.astype(np.float32)
    y_train_base = train_df["target"].values.astype(np.float32)

    X_test_base = test_df[base_feature_columns].fillna(0.0).values.astype(np.float32)
    y_test_base = test_df["target"].values.astype(np.float32)

    base_scaler = StandardScaler()
    X_train_base = base_scaler.fit_transform(X_train_base).astype(np.float32)
    X_test_base = base_scaler.transform(X_test_base).astype(np.float32)

    with open(BASE_SCALER_PATH, "wb") as f:
        pickle.dump(base_scaler, f)

    with open(BASE_FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(base_feature_columns, f)

    print("========== TRAINING BASE MODEL ==========")
    base_model, base_best_epoch, base_best_test_loss = train_model_with_resume(
        X_train_base,
        y_train_base,
        X_test_base,
        y_test_base,
        checkpoint_path=BASE_CHECKPOINT_PATH,
        best_model_path=BASE_BEST_MODEL_PATH,
        epochs=BASE_EPOCHS,
        lr=LR,
        hidden_dims=(128, 64),
        dropout=0.2,
        model_name="base_model",
    )

    print()
    print("Best base-model epoch     :", base_best_epoch)
    print(f"Best base-model test loss : {base_best_test_loss:.4f}")
    print()

    X_all_base = features_df[base_feature_columns].fillna(0.0).values.astype(np.float32)
    X_all_base = base_scaler.transform(X_all_base).astype(np.float32)
    X_all_base_tensor = torch.tensor(X_all_base, dtype=torch.float32)

    base_model.eval()
    with torch.no_grad():
        base_logits_all = base_model(X_all_base_tensor)
        base_probs_all = torch.sigmoid(base_logits_all).cpu().numpy()

    features_df["base_risk_score"] = base_probs_all

    print("Running community detection...")
    adjacency = build_undirected_adjacency(raw_data)
    partition = run_louvain_local_communities(
        adjacency=adjacency,
        max_passes=COMMUNITY_MAX_PASSES,
        gamma=COMMUNITY_GAMMA
    )

    base_risk_map = {}
    for row in features_df.itertuples(index=False):
        base_risk_map[int(row.txId)] = float(row.base_risk_score)

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

    community_id_col = []
    community_size_col = []
    community_hub_col = []
    community_auth_col = []
    community_hits_col = []
    community_refined_risk_col = []

    for row in features_df.itertuples(index=False):
        node = int(row.txId)
        cid = partition[node]

        community_id_col.append(cid)
        community_size_col.append(community_size_by_cid[cid])
        community_hub_col.append(community_hub_score[node])
        community_auth_col.append(community_authority_score[node])
        community_hits_col.append(community_hits_score[node])
        community_refined_risk_col.append(community_refined_risk[node])

    features_df["community_id"] = community_id_col
    features_df["community_size"] = community_size_col
    features_df["community_hub_score"] = community_hub_col
    features_df["community_authority_score"] = community_auth_col
    features_df["community_hits_score"] = community_hits_col
    features_df["community_refined_risk"] = community_refined_risk_col

    print("Community HITS features added.")
    print()

    labeled_df = features_df[features_df["label_binary"].notna()].copy()
    labeled_df["target"] = labeled_df["label_binary"].astype(int)

    train_df = labeled_df[labeled_df["txId"].astype(int).isin(train_txids)].copy()
    test_df = labeled_df[labeled_df["txId"].astype(int).isin(test_txids)].copy()

    final_feature_columns = [
        col for col in labeled_df.columns
        if col not in {
            "txId",
            "label",
            "label_binary",
            "target",
            "flow_uncertainty_score",
        }
    ]

    print("========== FINAL FEATURE COLUMNS ==========")
    for col in final_feature_columns:
        print(col)
    print()
    print("Number of final features:", len(final_feature_columns))
    print()

    X_train = train_df[final_feature_columns].fillna(0.0).values.astype(np.float32)
    y_train = train_df["target"].values.astype(np.float32)

    X_test = test_df[final_feature_columns].fillna(0.0).values.astype(np.float32)
    y_test = test_df["target"].values.astype(np.float32)

    final_scaler = StandardScaler()
    X_train = final_scaler.fit_transform(X_train).astype(np.float32)
    X_test = final_scaler.transform(X_test).astype(np.float32)

    with open(FINAL_SCALER_PATH, "wb") as f:
        pickle.dump(final_scaler, f)

    with open(FINAL_FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(final_feature_columns, f)

    print("========== TRAINING FINAL MODEL ==========")
    final_model, final_best_epoch, final_best_test_loss = train_model_with_resume(
        X_train,
        y_train,
        X_test,
        y_test,
        checkpoint_path=FINAL_CHECKPOINT_PATH,
        best_model_path=FINAL_BEST_MODEL_PATH,
        epochs=FINAL_EPOCHS,
        lr=LR,
        hidden_dims=(256, 128, 64),
        dropout=0.25,
        model_name="final_model",
    )

    print()
    print("Best final-model epoch     :", final_best_epoch)
    print(f"Best final-model test loss : {final_best_test_loss:.4f}")
    print()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    final_model.eval()
    with torch.no_grad():
        test_logits = final_model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    test_preds = (test_probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test.astype(int), test_preds)
    report = classification_report(y_test.astype(int), test_preds, digits=4)
    roc_auc = roc_auc_score(y_test.astype(int), test_probs)

    print("========== FINAL TEST RESULTS ==========")
    print("========== CONFUSION MATRIX ==========")
    print(cm)
    print()
    print("========== CLASSIFICATION REPORT ==========")
    print(report)
    print(f"ROC-AUC: {roc_auc:.4f}")

    features_df.to_csv(FINAL_FEATURES_OUTPUT_PATH, index=False)

    X_all_final = features_df[final_feature_columns].fillna(0.0).values.astype(np.float32)
    X_all_final = final_scaler.transform(X_all_final).astype(np.float32)
    X_all_final_tensor = torch.tensor(X_all_final, dtype=torch.float32)

    final_model.eval()
    with torch.no_grad():
        all_logits = final_model(X_all_final_tensor)
        all_probs = torch.sigmoid(all_logits).cpu().numpy()

    final_scores_df = pd.DataFrame({
        "txId": features_df["txId"].astype(int),
        "final_risk_score": all_probs
    })
    final_scores_df.to_csv(FINAL_SCORES_PATH, index=False)

    print()
    print("========== SAVED ARTIFACTS ==========")
    print("Base checkpoint      :", BASE_CHECKPOINT_PATH)
    print("Base best model      :", BASE_BEST_MODEL_PATH)
    print("Base scaler          :", BASE_SCALER_PATH)
    print("Base feature cols    :", BASE_FEATURE_COLS_PATH)
    print("Final checkpoint     :", FINAL_CHECKPOINT_PATH)
    print("Final best model     :", FINAL_BEST_MODEL_PATH)
    print("Final scaler         :", FINAL_SCALER_PATH)
    print("Final feature cols   :", FINAL_FEATURE_COLS_PATH)
    print("All features CSV     :", FINAL_FEATURES_OUTPUT_PATH)
    print("Final risk scores    :", FINAL_SCORES_PATH)
    print()
    print("Done.")
