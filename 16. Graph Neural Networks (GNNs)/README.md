# 16. Graph Neural Networks (GNNs)

This section develops graph neural networks from graph representation and message passing to Graph Convolutional Networks (GCN), GraphSAGE, Graph Attention Networks (GAT), graph classification, and link prediction.

The notebooks emphasize **how information moves through a graph**, how neighborhood aggregation differs across GNN architectures, and how graph structure is used together with node features for learning.

## Learning Progression

**Graph Representation** → **Message Passing** → **GCN** → **GraphSAGE** → **GAT** → **Graph Classification** → **Link Prediction**

## Notebooks

| # | Notebook | Main Methods / Focus |
|---:|---|---|
| 01 | `01_graph_fundamentals_and_message_passing.ipynb` | Graph representation, adjacency matrices, node features, neighborhoods, message passing, aggregation, manual calculations |
| 02 | `02_graph_convolutional_networks_gcn.ipynb` | Self-loops, degree normalization, GCN equation, full matrix calculation, GCN from scratch, Cora node classification |
| 03 | `03_graphsage_and_inductive_learning.ipynb` | Mean aggregation, concatenation, neighborhood sampling, inductive learning, GraphSAGE implementation |
| 04 | `04_graph_attention_networks_gat.ipynb` | Graph attention coefficients, masked neighborhood attention, multi-head attention, GAT implementation, attention inspection |
| 05 | `05_graph_classification_and_link_prediction.ipynb` | Graph pooling, graph classification, edge scoring, negative sampling, link prediction, ROC-AUC and Average Precision |

## What This Folder Demonstrates

- Representation of graphs using nodes, edges, adjacency matrices, and feature matrices
- Neighborhood aggregation and message passing
- Manual numerical calculations for GNN updates
- Matrix-based derivation of Graph Convolutional Networks
- GCN implementation without PyTorch Geometric
- Node classification with PyTorch Geometric
- Inductive neighborhood aggregation with GraphSAGE
- Attention-weighted message passing with GAT
- Graph-level prediction using global pooling
- Link prediction using learned node embeddings
- Evaluation of node, edge, and graph prediction tasks
- Comparison of GCN, GraphSAGE, and GAT

## Datasets

The notebooks use a combination of small educational graphs and benchmark graph datasets.

- **Toy graph** — used for transparent manual calculations
- **Cora** — used for node classification and link prediction
- **MUTAG** — used for graph classification

Benchmark datasets are downloaded automatically by PyTorch Geometric when the corresponding notebook is run. Large downloaded datasets are not committed to the repository.

## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- NetworkX
- PyTorch
- PyTorch Geometric
- scikit-learn
- Jupyter Notebook / Google Colab

## Repository Organization

```text
16. Graph Neural Networks (GNNs)/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_graph_fundamentals_and_message_passing.ipynb
├── 02_graph_convolutional_networks_gcn.ipynb
├── 03_graphsage_and_inductive_learning.ipynb
├── 04_graph_attention_networks_gat.ipynb
├── 05_graph_classification_and_link_prediction.ipynb
└── data/
    ├── README.md
    ├── toy_graph_nodes.csv
    └── toy_graph_edges.csv
```

## Visualization Approach

This folder does not include decorative architecture pictures. Visualizations are generated directly inside the notebooks only when they explain an actual computation or learned behavior, such as graph connectivity, neighborhood reach, learned embeddings, attention weights, learning curves, and link-score distributions.

## Focus

The folder shows how deep learning changes when data are connected by relationships rather than arranged only as vectors, grids, or sequences. The emphasis is on understanding message passing mathematically, verifying the operations with small numerical examples, and then applying the same ideas to real graph-learning tasks.
