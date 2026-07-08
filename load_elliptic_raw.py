from dataclasses import dataclass
from pathlib import Path
import zipfile
import pandas as pd
import networkx as nx


@dataclass
class EllipticRawData:
    node_ids: list
    node_to_idx: dict
    idx_to_node: dict
    timesteps: dict
    labels: dict
    edges: list
    in_neighbors: dict
    out_neighbors: dict
    graph: nx.DiGraph


def load_elliptic_raw(
    classes_path,
    edgelist_path,
    features_path
):

    classes_path = Path(classes_path)
    edgelist_path = Path(edgelist_path)
    features_path = Path(features_path)

   
    classes_df = pd.read_csv(classes_path)

    def map_label(x):
        x = str(x).strip().lower()
        if x == "1":
            return 1
        elif x == "2":
            return 0
        else:
            return -1

    classes_df["label"] = classes_df["class"].apply(map_label)

    labels = dict(zip(classes_df["txId"], classes_df["label"]))

   
    if features_path.suffix == ".zip":
        with zipfile.ZipFile(features_path, "r") as zf:
            csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
            if len(csv_names) == 0:
                raise ValueError("No CSV found inside features zip.")
            features_df = pd.read_csv(zf.open(csv_names[0]), header=None)
    else:
        features_df = pd.read_csv(features_path, header=None)

   
    features_df = features_df.iloc[:, :2].copy()
    features_df.columns = ["txId", "timestep"]

    timesteps = dict(zip(features_df["txId"], features_df["timestep"]))

   
    edges_df = pd.read_csv(edgelist_path)

    edges = list(zip(edges_df["txId1"], edges_df["txId2"]))

   
    valid_nodes = set(labels.keys()) & set(timesteps.keys())

    filtered_edges = []
    for src, dst in edges:
        if src in valid_nodes and dst in valid_nodes:
            filtered_edges.append((src, dst))

    
    graph_nodes = set()
    for src, dst in filtered_edges:
        graph_nodes.add(src)
        graph_nodes.add(dst)

    final_nodes = sorted(valid_nodes | graph_nodes)

   
    G = nx.DiGraph()
    G.add_nodes_from(final_nodes)
    G.add_edges_from(filtered_edges)

  
    in_neighbors = {}
    out_neighbors = {}

    for node in G.nodes():
        in_neighbors[node] = list(G.predecessors(node))
        out_neighbors[node] = list(G.successors(node))

 
    node_ids = list(G.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    idx_to_node = {idx: node_id for idx, node_id in enumerate(node_ids)}


    labels = {node: labels.get(node, -1) for node in node_ids}
    timesteps = {node: timesteps[node] for node in node_ids}

    return EllipticRawData(
        node_ids=node_ids,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        timesteps=timesteps,
        labels=labels,
        edges=filtered_edges,
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
        graph=G
    )


if __name__ == "__main__":
    data = load_elliptic_raw(
        classes_path="data/elliptic_txs_classes.csv",
        edgelist_path="data/elliptic_txs_edgelist.csv",
        features_path="data/elliptic_txs_features.csv"
    )

    print("========== RAW ELLIPTIC DATA LOADED ==========")
    print(f"Number of nodes: {len(data.node_ids)}")
    print(f"Number of edges: {len(data.edges)}")

    num_licit = sum(1 for x in data.labels.values() if x == 0)
    num_illicit = sum(1 for x in data.labels.values() if x == 1)
    num_unknown = sum(1 for x in data.labels.values() if x == -1)

    print(f"Licit nodes:   {num_licit}")
    print(f"Illicit nodes: {num_illicit}")
    print(f"Unknown nodes: {num_unknown}")

    timesteps = list(data.timesteps.values())
    print(f"Timestep range: {min(timesteps)} to {max(timesteps)}")
