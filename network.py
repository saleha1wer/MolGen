import torch
from torch import nn
from mol2graph import Graphs

class GNN(nn.Module):
    def __init__(self, node_feature_dimension, num_propagation_steps :int =4):
        super().__init__()

        self.num_propagation_steps = num_propagation_steps
        # called T above.

        # Our sub modules:
        self.message_projection = nn.Linear(node_feature_dimension, node_feature_dimension, bias=False)
        self.update_gru = nn.GRUCell(input_size=node_feature_dimension,
                                     hidden_size=node_feature_dimension, bias=True)
        self.attn_net = nn.Linear(node_feature_dimension, 1)
        self.proj_net = nn.Linear(node_feature_dimension, node_feature_dimension)
        self.final_lin = nn.Linear(node_feature_dimension, 1)

    def forward(self, graphs_in: Graphs):
        """
        Produces a column vector of predictions, with each element in this vector a prediction
        for each marked graph in `graphs_in`.

        In the comments below N is the number of nodes in graph_in (across all graphs),
        d the feature dimension, and G is the number of individual molecular graphs.
        """
        # 1. Message passing and updating
        m = graphs_in.node_features  # shape: [N, d]

        for t in range(self.num_propagation_steps):
            projs = self.message_projection(m)  # [N, d]

            # Update the node embeddings (eqn 1 above)
            # 1a. compute the sum for each node
            msgs = torch.zeros_like(m)  # [N, d]
            msgs.index_add_(0, graphs_in.edge_list[:, 0], projs.index_select(0, graphs_in.edge_list[:, 1]))

            # 1b. update the embeddings via GRU cell
            m = self.update_gru(msgs, m)  # [N, d]

        # 2. Aggregation (eqn 2 above)
        # a compute weighted embeddings
        attn_coeffs = torch.sigmoid(self.attn_net(m))  # [N, 1]
        proj_embeddings = self.proj_net(m)  # [N, d']
        weighted_embeddings = attn_coeffs * proj_embeddings

        # perform the sum
        graph_embedding = torch.zeros(graphs_in.num_graphs, weighted_embeddings.shape[1],
                                      device=m.device, dtype=m.dtype)
        graph_embedding.index_add_(0, graphs_in.node_to_graph_id, weighted_embeddings)  # [G, d']

        # 3. Final linear projection.
        final_prediction = self.final_lin(graph_embedding)  # [G, 1]
        return final_prediction