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
from torch_geometric.data import Data


class GNN(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(GNN, self).__init__()

        self.data_dir = data_dir or os.getcwd()  # pass this from now on

        self.learning_rate = config['learning_rate']
        self.node_feature_dim = config['node_feature_dimension']
        self.edge_dim = config['edge_feature_dimension']
        self.hidden_size = config['embedding_dimension']
        self.propagation_steps = config['num_propagation_steps']
        self.num_heads = 1

        self.gat1 = GATConv(in_channels=self.node_feature_dim, out_channels=self.hidden_size * self.num_heads,
                            heads=self.num_heads, edge_dim=self.edge_dim, aggr='add')
        self.gat2 = GATConv(in_channels=self.hidden_size, out_channels=self.hidden_size * self.num_heads,
                            heads=self.num_heads, edge_dim=self.edge_dim, aggr='add')

        self.final_lin = nn.Linear(self.hidden_size * self.num_heads, 1)

    def forward(self, graphs: Data):
        """
        Produces a column vector of predictions, with each element in this vector a prediction
        for each marked graph in `graphs_in`.

        In the comments below N is the number of nodes in graph_in (across all graphs),
        d the feature dimension, and G is the number of individual molecular graphs.
        d' is the embedding dimension
        """

        # x, edge_attr = graphs.x.float(), graphs.edge_attr.float()
        # graphs.edge_index = graphs.edge_index.t().to(torch.long)
        graph_embedding = self.gat1(x=graphs.x, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr)  # [N, d']

        for _ in range(self.propagation_steps):
            graph_embedding = self.gat2(x=graph_embedding, edge_index=graphs.edge_index)

        graph_embedding = global_mean_pool(graph_embedding, graphs.batch)  # [G, d']

        # graph_embedding = F.relu(graph_embedding)

        # 3. Final linear projection.
        final_prediction = self.final_lin(graph_embedding)  # [G, 1]
        return final_prediction

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