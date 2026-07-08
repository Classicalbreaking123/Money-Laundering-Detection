# Money Laundering Detection using Graph Learning, Community Refinement, and Flow Uncertainty

## Overview

This project tackles money laundering detection on transaction graphs using a combination of graph-structural features, a base risk model, community detection, community-level risk refinement, and a custom uncertainty / information-theoretic feature that measures how diffusely money can flow outward from a transaction.

The dataset is modeled as a directed transaction graph:

- each node represents a transaction
- each directed edge represents flow of money from one transaction to another

The goal is to assign each transaction a risk score and classify it as licit or illicit.

---

## Why graph-based laundering detection?

Money laundering is rarely just one suspicious transaction in isolation. It is usually a network phenomenon:

- funds are split
- funds are merged
- money is passed through relay accounts
- transactions form dense communities
- flow is dispersed to create uncertainty about origin
- funds are routed through sequences of individually harmless-looking transactions

So instead of treating laundering as a purely tabular classification problem, this project models it as a graph problem and tries to capture three things:

1. Local transaction structure  
   Is a transaction mostly receiving, mostly sending, or acting as a pass-through relay?

2. Community structure  
   Does it sit inside a suspicious ring / laundering cluster?

3. Flow uncertainty  
   Once money reaches this node, how many plausible downstream routes does it have? Is the outflow tightly concentrated or deliberately diffused?

---

## Project pipeline

The pipeline has three stages.

### Stage 1 — Build graph features

From the transaction graph, I construct a set of hand-designed graph features. These are meant to capture laundering behaviors such as:

- splitting funds into many downstream branches
- merging funds from many upstream sources
- pass-through relay behavior
- belonging to cycles / tightly connected components
- bridging between weakly connected regions
- temporal burst patterns
- uncertain / diffuse downstream flow

The main feature families are:

### Degree / flow-shape features
- indegree
- outdegree
- degree_imbalance
- pass_through
- split_score

### Structural features
- scc_size
- core_number
- clustering_coeff

### Bridge / relay features
- bridge_relay_score
- bridge_cross_density
- bridge_score

### Temporal features
- temporal_in_degree
- temporal_out_degree
- temporal_pass_through
- temporal_burst_entropy_score

### Uncertainty / information-theoretic feature
- flow_uncertainty_score

This last one is one of the most interesting custom features in the project, so it is explained separately below.

### Stage 2 — Train a base transaction risk model

Using the graph features from Stage 1, I train a base MLP classifier that outputs a transaction-level risk score or illicit probability.

So for each transaction i, the model produces a value like:

p(i) = probability that transaction i is illicit

A high value means the node looks suspicious. A low value means it looks licit.

This base score is important because the next stage performs community-level refinement on top of it.

So the base model answers:

“If I only look at this transaction and its graph-derived features, how suspicious does it look?”

### Stage 3 — Community detection + risk refinement

Money laundering often happens in groups of related transactions, not just isolated nodes.

So after obtaining the base risk score for every transaction, I run community detection on the graph to find transaction clusters. Inside each detected community, I then refine risk by letting suspiciousness interact with who sends to whom inside the community.

This produces a second-stage score that is no longer purely local.

---

# Key ideas in the project

## 1) Graph features designed for laundering behavior

The goal of the graph features is not to throw every standard graph metric at the problem. The point is to encode behaviors that are actually meaningful for laundering.

### 1.1 Degree imbalance

Formula:

degree_imbalance(i) = (outdegree(i) - indegree(i)) / (outdegree(i) + indegree(i) + eps)

Interpretation:

- near +1 -> mostly sending
- near -1 -> mostly receiving
- near 0 -> balanced inflow / outflow

This helps distinguish collectors, distributors, and relay nodes.

### 1.2 Pass-through score

Formula:

pass_through(i) = min(indegree(i), outdegree(i)) / (max(indegree(i), outdegree(i)) + eps)

This is high when a node has both incoming and outgoing flow in a balanced way.

So it behaves like a relay / transit node rather than a pure sink or pure source. That matters because laundering chains often use intermediate nodes whose purpose is not to keep money, but to receive and quickly forward it.

### 1.3 Split score

Formula:

split_score(i) = outdegree(i) / (indegree(i) + 1)

A high split score means one incoming stream is being dispersed into many outgoing branches. That is a classic laundering behavior because splitting makes tracing the money harder.

### 1.4 SCC size

If a node belongs to a large strongly connected component, then it lies inside a region where transactions are mutually reachable by directed paths.

That is interesting because laundering rings often reuse funds in cyclic or tightly interlinked structures.

### 1.5 Core number

The core number of a node tells us how deep it sits inside a dense part of the graph.

Intuition:

- low core number -> peripheral node
- high core number -> structurally embedded inside a dense transaction region

If a laundering operation uses a network of repeatedly interacting accounts or transactions, those nodes often lie deeper in the graph core rather than on the periphery.

### 1.6 Bridge score

A transaction can also be suspicious if it acts as a bridge between two otherwise weakly connected regions.

I compute bridge features by looking at the node’s incoming-side neighborhood and outgoing-side neighborhood and asking:

- how much the node behaves like a relay between those sides
- how much those two sides are already directly connected without the node

The bridge score is high when the node is an important connector / bottleneck rather than a redundant middleman.

---

## 2) Temporal Burst Entropy

Simple burst-count features only count how many neighbors occur near the same time. That is useful, but it misses how concentrated the timing pattern is.

So instead I use a more structured temporal feature.

For a node i, take all neighboring transactions and look at their timestep offsets relative to node i:

delta_t = timestep(neighbor) - timestep(i)

Now form the empirical distribution over these offsets. If almost all neighbors happen in the same tiny time window, the distribution is highly concentrated. If neighbors are spread across many different offsets, the distribution is diffuse.

Let p(delta) be the fraction of neighbors with offset delta. Then the entropy is:

H(i) = - sum over delta of p(delta) * log(p(delta))

I then convert it into a concentration-style score:

temporal_burst_entropy_score(i) = 1 - H(i) / log(K(i))

where K(i) is the number of distinct observed offsets.

Interpretation:

- high score -> the node’s neighborhood is temporally concentrated / bursty
- low score -> the node’s interactions are spread across time

This is useful because suspicious transaction chains are often executed in short, coordinated bursts.

---

## 3) Flow Uncertainty Score (Discounted Flow Entropy Rate)

This is the most information-theoretic feature in the project.

### Motivation

One laundering objective is to make it hard to tell where the money ultimately goes. That often means deliberately creating many plausible downstream routes.

So for each transaction node i, I want to measure:

“If money starts at node i, how much uncertainty is there in its downstream flow paths?”

Not just immediate branching at one hop, but multi-hop downstream uncertainty.

### 3.1 Step 1 — Define a transition distribution over outgoing edges

Suppose node i has outgoing neighbors j.

I assign a transition weight to each outgoing neighbor j:

w(i, j) = 1 + alpha * outdegree(j) + gamma * two_hop_reach(j)

Then normalize:

p(i, j) = w(i, j) / sum over outgoing neighbors k of w(i, k)

So p(i, j) is the probability that flow from node i continues to child j.

#### Intuition

A child is given more weight if it can itself spread money further:

- large outdegree = many immediate next routes
- large two-hop reach = access to a broader downstream region

So the transition model favors children that are structurally better at further dispersing flow.

### 3.2 Step 2 — Compute one-step flow entropy

At node i, the one-step uncertainty is the entropy of the outgoing transition distribution:

h1(i) = - sum over outgoing neighbors j of p(i, j) * log(p(i, j))

This captures the uncertainty in where money goes in the next step.

But laundering is not only about one hop. A node may have only a few outgoing edges now, but those edges may open into a large uncertain downstream tree.

So we need a recursive multi-hop version.

### 3.3 Step 3 — Discounted downstream uncertainty recurrence

Define F(i) = total downstream flow uncertainty starting from node i.

Then:

F(i) = h1(i) + beta * sum over outgoing neighbors j of p(i, j) * F(j)

This says:

- first pay the local uncertainty at node i
- then add the expected future uncertainty of the children
- but discount future hops by beta, where 0 < beta < 1

So:

- if beta is small -> mostly immediate branching matters
- if beta is large -> long downstream laundering chains matter more

This is exactly the feature called:

flow_uncertainty_score(i) = F(i)

### 3.4 Why this is useful

A high flow uncertainty score means:

- money leaving this transaction has many plausible downstream routes
- those routes themselves keep branching or spreading
- attribution of the final destination becomes harder

So this feature is trying to quantify the laundering intuition:

“This node is a good place to hide the true flow of funds because once money reaches it, the downstream path becomes uncertain.”

---

## 4) Base risk model

After building the graph features, I train a neural network that maps transaction features to a base risk score.

So for each transaction i, the network produces:

b(i) = estimated probability that transaction i is illicit

This score is purely transaction-level. It does not yet know about community-level collective laundering behavior.

So b(i) is the first approximation of suspiciousness.

The final licit / illicit prediction is then obtained by thresholding this score.

For example:

- if b(i) >= threshold, predict illicit
- else predict licit

---

## 5) Community detection

Once each node has a base risk score, I move to a second level of reasoning:

“Even if a transaction is only moderately suspicious by itself, does it sit inside a suspicious group of tightly interacting transactions?”

To do that, I run community detection on the graph.

The implementation uses a Louvain-style modularity optimization on the undirected version of the transaction graph. The idea is to partition the graph into communities so that nodes are more densely connected within communities than across them.

Why this matters for laundering:

- laundering often uses clusters of cooperating accounts / transactions
- suspiciousness should not be judged only node-by-node
- a node surrounded by suspicious peers in the same transaction ring deserves extra scrutiny

---

## 6) Community-level HITS refinement

This is the most custom part of the project after the flow uncertainty feature.

The question here is:

“Once I already have a base risk score b(i), how do I refine it using the directed structure inside its community?”

Instead of using only a simple community-average risk, I run a risk-aware HITS-style propagation inside each community.

### 6.1 Why HITS?

HITS normally assigns two scores:

- hub score — points to important nodes
- authority score — receives links from important hubs

In a transaction graph, this is interesting because inside a suspicious community:

- a hub-like transaction may disperse money toward risky downstream nodes
- an authority-like transaction may receive money from risky upstream relays

So HITS gives a way to distinguish different roles inside the laundering subgraph.

### 6.2 Risk-aware HITS equations

Inside a community, let b(i) be the base risk of node i.

I update authority and hub scores iteratively.

Authority update:

a(i) = sum over nodes j -> i of b(j) * h(j)

Hub update:

h(i) = sum over nodes i -> k of b(k) * a(k)

Then I normalize after each step.

Interpretation:

- a node gets high authority if risky hubs point into it
- a node gets high hub if it points to risky authorities

This means community structure and base suspiciousness reinforce each other.

### 6.3 Turn HITS into a community suspiciousness score

Once hub and authority stabilize, I combine them:

c(i) = (h(i) + a(i)) / 2

Then I min-max normalize inside the community so that c(i) becomes a community-local suspiciousness score.

### 6.4 Final refined community risk

Finally I blend the original base risk with the community HITS score:

r(i) = (b(i) + lambda * c(i)) / (1 + lambda)

where:

- b(i) = base risk from the first-stage model
- c(i) = suspiciousness induced by the transaction’s role inside its community
- lambda = how strongly to trust the community refinement

So the final community-refined risk is not replacing the base model. It is saying:

“Take the transaction’s own suspiciousness, then adjust it based on how suspiciously it behaves inside its transaction community.”

---

## Why this two-stage setup is useful

A single node-level model can miss group effects.

For example, a transaction may not look obviously illicit in isolation, but:

- it lies in a dense suspicious community
- it receives from risky upstream nodes
- it forwards to risky downstream nodes
- it occupies a hub / authority role inside that laundering cluster

The community refinement stage is designed to capture exactly this.

So the model is doing:

### Stage A — Node-level reasoning
Does this transaction itself look suspicious from its local graph features?

### Stage B — Community-level reasoning
Given where this transaction sits inside a suspicious subgraph, should its risk be revised upward or downward?

---

## Final model

After computing:

- base graph features
- base risk score
- community-level HITS refinement features

I train a final neural model on the enriched feature set.

So the final classifier sees both:

1. raw graph-derived laundering signals
2. community-refined risk signals

and outputs a final risk score for each transaction. That score is then thresholded to classify the transaction as licit or illicit.

---

## Most important custom pieces in the project

If I had to summarize the most important / non-trivial parts of the project, they would be:

### 1. Flow Uncertainty Score
A discounted entropy-rate style feature that measures how uncertain downstream money flow becomes after a transaction.

### 2. Temporal Burst Entropy
A temporal concentration feature based on entropy of neighbor time offsets, instead of just counting bursts.

### 3. Bridge / Relay Features
Designed to identify transactions that act as important connectors between two transaction regions.

### 4. Community HITS Refinement
Instead of using only average community risk, I refine node risk by modeling hub / authority roles inside the community, weighted by base suspiciousness.

That combination is what makes the project more than “just train a classifier on transaction data”.

---

## Repository structure

A minimal setup is:

Money-Laundering-Detection/
- README.md
- train_all_in_one_money_laundering.py
- load_elliptic_raw.py
- data/
  - elliptic_txs_classes.csv
  - elliptic_txs_edgelist.csv
  - elliptic_txs_features.csv

If you do not want to upload large artifacts or raw data to GitHub, keep only the code and README in the repo, and place the Elliptic dataset manually inside the data/ folder before running the project.

---

## Running the project

Run the full pipeline with:

python train_all_in_one_money_laundering.py

This script performs:

1. raw data loading
2. graph construction
3. graph feature computation
4. base risk model training
5. community detection
6. community-level HITS refinement
7. final model training
8. saving outputs / checkpoints

---

## Dataset note

The repository may not include the raw Elliptic dataset files because they are large.

To run the project, place the following files inside the data/ folder:

- elliptic_txs_classes.csv
- elliptic_txs_edgelist.csv
- elliptic_txs_features.csv

---

## Takeaway

The core idea of this project is:

Money laundering is fundamentally a graph + flow + community problem.

So instead of treating transactions independently, the model tries to capture:

- how funds are routed
- how uncertainty is created downstream
- how suspicious communities reinforce risk
- how a transaction’s role inside a laundering subgraph changes its final score

That is why the project combines:

- graph feature engineering
- information-theoretic flow uncertainty
- community detection
- community-level risk propagation
