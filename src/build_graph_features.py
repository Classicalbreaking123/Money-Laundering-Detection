from pathlib import Path
import heapq
import math
import pandas as pd
import networkx as nx

from load_elliptic_raw import load_elliptic_raw


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


def compute_hits_manual(G, max_iter=100, tol=1e-8):
    nodes = list(G.nodes())

    in_neighbors = {}
    out_neighbors = {}

    for node in nodes:
        in_neighbors[node] = list(G.predecessors(node))
        out_neighbors[node] = list(G.successors(node))

    hub = {}
    authority = {}

    for node in nodes:
        hub[node] = 1.0
        authority[node] = 1.0

    for _ in range(max_iter):
        new_authority = {}
        for node in nodes:
            score = 0.0
            for nbr in in_neighbors[node]:
                score = score + hub[nbr]
            new_authority[node] = score

        auth_norm = math.sqrt(sum(val * val for val in new_authority.values()))
        if auth_norm > 0:
            for node in nodes:
                new_authority[node] = new_authority[node] / auth_norm

        new_hub = {}
        for node in nodes:
            score = 0.0
            for nbr in out_neighbors[node]:
                score = score + new_authority[nbr]
            new_hub[node] = score

        hub_norm = math.sqrt(sum(val * val for val in new_hub.values()))
        if hub_norm > 0:
            for node in nodes:
                new_hub[node] = new_hub[node] / hub_norm

        diff = 0.0
        for node in nodes:
            diff = diff + abs(new_authority[node] - authority[node])
            diff = diff + abs(new_hub[node] - hub[node])

        authority = new_authority
        hub = new_hub

        if diff < tol:
            break

    return hub, authority


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


def compute_graph_features(raw_data):
    G = raw_data.graph
    node_ids = raw_data.node_ids
    timesteps = raw_data.timesteps

    indegree = dict(G.in_degree())
    outdegree = dict(G.out_degree())
    pagerank = nx.pagerank(G, alpha=0.85)

    hub_score, authority_score = compute_hits_manual(G)

    scc_size = {}
    sccs = list(nx.strongly_connected_components(G))
    for comp in sccs:
        size = len(comp)
        for node in comp:
            scc_size[node] = size

    G_und = G.to_undirected()

    core_number = compute_core_numbers_manual(G_und)

    wcc_size = {}
    wccs = list(nx.connected_components(G_und))
    for comp in wccs:
        size = len(comp)
        for node in comp:
            wcc_size[node] = size

    clustering_coeff = nx.clustering(G_und)

    bridge_relay_score, bridge_cross_density, bridge_score = compute_bridge_features(
        raw_data, G_und, node_ids
    )

    total_degree = {}
    for node in node_ids:
        total_degree[node] = indegree.get(node, 0) + outdegree.get(node, 0)

    avg_neighbor_degree = {}
    two_hop_reach = {}
    temporal_burst = {}
    temporal_in_degree = {}
    temporal_out_degree = {}

    for node in node_ids:
        in_neighbors = set(raw_data.in_neighbors[node])
        out_neighbors = set(raw_data.out_neighbors[node])
        neighbors = in_neighbors | out_neighbors

        if len(neighbors) == 0:
            avg_neighbor_degree[node] = 0.0
        else:
            avg_neighbor_degree[node] = sum(total_degree[nbr] for nbr in neighbors) / len(neighbors)

        two_hop_nodes = set()
        for nbr in neighbors:
            nbr_neighbors = set(raw_data.in_neighbors[nbr]) | set(raw_data.out_neighbors[nbr])
            two_hop_nodes.update(nbr_neighbors)
        if node in two_hop_nodes:
            two_hop_nodes.remove(node)
        two_hop_reach[node] = len(two_hop_nodes)

        t = timesteps[node]

        burst_count = 0
        for nbr in neighbors:
            if abs(timesteps[nbr] - t) <= 1:
                burst_count = burst_count + 1
        temporal_burst[node] = burst_count

        temp_in = 0
        for nbr in in_neighbors:
            if abs(timesteps[nbr] - t) <= 1:
                temp_in = temp_in + 1
        temporal_in_degree[node] = temp_in

        temp_out = 0
        for nbr in out_neighbors:
            if abs(timesteps[nbr] - t) <= 1:
                temp_out = temp_out + 1
        temporal_out_degree[node] = temp_out

    rows = []

    for node in node_ids:
        in_deg = indegree.get(node, 0)
        out_deg = outdegree.get(node, 0)

        degree_imbalance = (out_deg - in_deg) / (out_deg + in_deg + 1e-9)
        pass_through = min(in_deg, out_deg) / (max(in_deg, out_deg) + 1e-9)
        split_score = out_deg / (in_deg + 1.0)
        merge_score = in_deg / (out_deg + 1.0)
        cycle_flag = 1 if scc_size.get(node, 1) > 1 else 0

        temp_in = temporal_in_degree[node]
        temp_out = temporal_out_degree[node]
        temporal_pass_through = min(temp_in, temp_out) / (max(temp_in, temp_out) + 1e-9)

        row = {
            "txId": node,
            "timestep": raw_data.timesteps[node],
            "label": raw_data.labels[node],
            "indegree": in_deg,
            "outdegree": out_deg,
            "degree_imbalance": degree_imbalance,
            "pass_through": pass_through,
            "split_score": split_score,
            "merge_score": merge_score,
            "pagerank": pagerank.get(node, 0.0),
            "hub_score": hub_score.get(node, 0.0),
            "authority_score": authority_score.get(node, 0.0),
            "scc_size": scc_size.get(node, 1),
            "wcc_size": wcc_size.get(node, 1),
            "cycle_flag": cycle_flag,
            "core_number": core_number.get(node, 0),
            "two_hop_reach": two_hop_reach[node],
            "avg_neighbor_degree": avg_neighbor_degree[node],
            "temporal_burst": temporal_burst[node],
            "temporal_in_degree": temp_in,
            "temporal_out_degree": temp_out,
            "temporal_pass_through": temporal_pass_through,
            "clustering_coeff": clustering_coeff.get(node, 0.0),
            "bridge_relay_score": bridge_relay_score.get(node, 0.0),
            "bridge_cross_density": bridge_cross_density.get(node, 1.0),
            "bridge_score": bridge_score.get(node, 0.0),
        }
        rows.append(row)

    features_df = pd.DataFrame(rows)
    return features_df


if __name__ == "__main__":
    raw_data = load_elliptic_raw(
        classes_path="data/elliptic_txs_classes.csv",
        edgelist_path="data/elliptic_txs_edgelist.csv",
        features_path="data/elliptic_txs_features.csv"
    )

    features_df = compute_graph_features(raw_data)

    print("========== GRAPH FEATURES ==========")
    print(features_df.head())
    print()
    print("Shape:", features_df.shape)
    print()
    print("Label counts:")
    print(features_df["label"].value_counts(dropna=False).sort_index())

    output_path = Path("data/graph_features.csv")
    features_df.to_csv(output_path, index=False)
    print()
    print(f"Saved features to: {output_path}")
