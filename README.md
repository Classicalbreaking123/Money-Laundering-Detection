# Money Laundering Detection using Graph Features, Community Refinement, and Flow Uncertainty

## Overview

This project tackles **money laundering detection on transaction graphs** using a combination of:

- **graph-structural features** built directly from the transaction network,
- a **base risk model** that learns suspiciousness at the transaction level,
- **community detection** to identify tightly interacting transaction groups,
- **community-level risk refinement** to propagate suspicion inside laundering rings,
- and a custom **uncertainty / information-theoretic feature** that measures how diffusely money can flow outward from a transaction.

The dataset is modeled as a **directed transaction graph**:

- each node represents a transaction,
- each directed edge represents flow of money from one transaction to another.

The goal is to assign each transaction a **risk score** and classify it as **licit vs illicit**.

---

## Why graph-based laundering detection?

Money laundering is rarely just one suspicious transaction in isolation. It is usually a **network phenomenon**:

- funds are **split**,
- **merged**,
- passed through **relay accounts**,
- moved through **dense communities**,
- dispersed to create **uncertainty about origin**,
- and routed through sequences of transactions that individually look harmless.

So instead of treating laundering as a purely tabular classification problem, this project models it as a **graph learning problem** and tries to capture three things:

1. **Local transaction structure**  
   Is a transaction mostly receiving, mostly sending, or acting as a pass-through relay?

2. **Meso-scale community structure**  
   Does it sit inside a suspicious transaction cluster or laundering ring?

3. **Flow uncertainty**  
   Once money reaches this node, how many plausible downstream routes does it have?  
   Is the outflow tightly concentrated or deliberately diffused?

---

## Project pipeline

The pipeline has **three stages**.

### Stage 1 — Build graph features

From the transaction graph, I construct a set of **hand-designed graph features** meant to capture laundering behavior such as:

- **splitting** funds into many downstream branches,
- **pass-through relay behavior**,
- **belonging to tightly connected regions**,
- **bridging between weakly connected transaction regions**,
- **temporal burst patterns**,
- and **uncertain / diffuse downstream flow**.

The main feature families are:

#### Degree / flow-shape features
- `indegree`
- `outdegree`
- `degree_imbalance`
- `pass_through`
- `split_score`

#### Structural features
- `scc_size` — size of the strongly connected component
- `core_number` — how deeply embedded the node is in the graph core
- `clustering_coeff`

#### Bridge / relay features
- `bridge_relay_score`
- `bridge_cross_density`
- `bridge_score`

#### Temporal features
- `temporal_in_degree`
- `temporal_out_degree`
- `temporal_pass_through`
- `temporal_burst_entropy_score`

#### Uncertainty / information-theoretic feature
- `flow_uncertainty_score`

This last one is one of the most interesting custom features in the project, so it is explained separately below.

---

### Stage 2 — Train a base transaction risk model

Using the graph features from Stage 1, I train a **base MLP classifier** that outputs a transaction-level probability:

`b(i) in [0, 1]`

where:

- `b(i)` close to 1 means transaction `i` looks illicit,
- `b(i)` close to 0 means transaction `i` looks licit.

This base score is important because the next stage performs **community-level refinement** on top of it.

So the base model answers:

> If I only look at this transaction and its graph-derived features, how suspicious does it look?

---

### Stage 3 — Community detection + risk refinement

Money laundering often happens in **groups of related transactions**, not just isolated nodes.

So after obtaining the base risk score `b(i)` for every transaction, I run **community detection** on the graph to find transaction clusters. Inside each detected community, I then refine risk by letting suspiciousness interact with **who sends to whom inside the community**.

This produces a second-stage score that is no longer purely local.

---

# Key ideas in the project

## 1) Graph features designed for laundering behavior

The goal of the graph features is not to throw every standard graph metric at the problem. The point is to encode behaviors that are actually meaningful for laundering.

---

## 1.1 Degree imbalance

Formula:

```text
degree_imbalance(i) =
(outdegree(i) - indegree(i)) / (outdegree(i) + indegree(i) + eps)
