import torch
import os
import typing
import numpy as np
import time
import pandas as pd
import pytorch_lightning as pl
from dataclasses import dataclass

from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch_geometric.nn import TransformerConv, TopKPooling, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.models import GAT
from torch_geometric.data import Data

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GNN(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(GNN, self).__init__()

        self.data_dir = data_dir or os.getcwd()  # pass this from now on

        self.learning_rate = 0.01  #config['learning_rate']
        self.node_feature_dim = config['node_feature_dimension']
        self.edge_dim = config['edge_feature_dimension']
        self.hidden_size = config['embedding_dimension']
        self.propagation_steps = config['num_propagation_steps']

        embedding_size = self.hidden_size
        n_heads = 8
        self.n_layers = self.propagation_steps
        dropout_rate = 0.1
        top_k_ratio = 0.5
        self.top_k_every_n = 1
        dense_neurons = 256
        edge_dim = 3

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(self.node_feature_dim,
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)

        self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        # Linear layers
        self.linear1 = Linear(embedding_size * 2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = Linear(int(dense_neurons / 2), 1)

    # self.gat = GAT(self.node_feature_dim, hidden_channels=64, num_layers=4, out_channels=self.hidden_size,
    #                    edge_dim=self.edge_dim)

        # self.num_heads = 8
        #
        # self.gat1 = GATConv(in_channels=self.node_feature_dim, out_channels=self.hidden_size,
        #                     heads=self.num_heads, edge_dim=self.edge_dim, aggr='add')
        # # self.gat2 = GATConv(in_channels=self.hidden_size, out_channels=self.hidden_size * self.num_heads,
        # #                     heads=self.num_heads, edge_dim=self.edge_dim, aggr='add')
        #
        # self.final_lin = nn.Linear(self.hidden_size, 1)

    def forward(self, graphs):
        # Initial transformation
        x, edge_attr, edge_index, batch_index = graphs.x, graphs.edge_attr, graphs.edge_index, graphs.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                )
                # Add current representation
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x

    # def forward(self, graphs: Data):
    #     """
    #     Produces a column vector of predictions, with each element in this vector a prediction
    #     for each marked graph in `graphs_in`.
    #
    #     In the comments below N is the number of nodes in graph_in (across all graphs),
    #     d the feature dimension, and G is the number of individual molecular graphs.
    #     d' is the embedding dimension
    #     """
    #
    #     # x, edge_attr = graphs.x.float(), graphs.edge_attr.float()
    #     # graphs.edge_index = graphs.edge_index.t().to(torch.long)
    #     # for _ in range(self.propagation_steps):
    #     #     graph_embedding = self.gat1.propagate(edge_index=graphs.edge_index)
    #     #
    #     # graph_embedding = self.gat1(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr)  # [N, d']
    #     #
    #     # graph_embedding = F.relu(graph_embedding)
    #     #
    #     #
    #
    #     # graph_embedding = F.relu(graph_embedding)
    #     graph_embedding = self.gat(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr)  # [N, d, e]
    #     graph_embedding = global_mean_pool(graph_embedding, graphs.batch)  # [G, d']
    #
    #     # 3. Final linear projection.
    #     final_prediction = self.final_lin(graph_embedding)  # [G, 1]
    #     return final_prediction

    def mse_loss(self, prediction, target):
        result = F.mse_loss(prediction, target)
        return result

    def training_step(self,train_batch, batch_idx):
        # print(f'printing x:{train_batch.x}')
        # print(f'shape of y:{train_batch.y.shape}') # TODO continue here, for some reason only 1 graph gets passed in but a vector of multiple y (target) values, is this correct???
        prediction = self.forward(train_batch)
        # print(f'printing prediction:{prediction}')
        loss = self.mse_loss(prediction, train_batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        prediction = self.forward(val_batch)
        loss = self.mse_loss(prediction, val_batch.y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def plot_train_and_val_using_mpl(train_loss, val_loss, name=None, save=False):
    """
    Plots the train and validation loss using Matplotlib
    """
    assert len(train_loss) == len(val_loss)

    f, ax = plt.subplots()
    x = np.arange(len(train_loss))
    ax.plot(x, np.array(train_loss), label='train')
    ax.plot(x, np.array(val_loss), label='val')
    ax.legend()
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    if save:
        if name != None:
            ax.set_title(name)
        else:
            ax.set_title('Anonymous plot')
        plt.savefig(name + '.png')
    return f, ax