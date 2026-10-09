# Data

This folder contains only the small toy graph used for transparent calculations.

The larger benchmark datasets used in the notebooks are downloaded automatically by PyTorch Geometric:

- Cora for node classification and link prediction
- MUTAG for graph classification

Downloaded benchmark files are excluded from version control through `.gitignore`.

## Toy Graph Files

### `toy_graph_nodes.csv`

Contains one row per node with two numerical node features.

### `toy_graph_edges.csv`

Contains the undirected edge list used in the introductory message-passing calculations.
