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
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import Data

from utils.from_smiles import GraphRegressionDataset, Graphs


@dataclass
class TrainParams:
    batch_size: int = 64
    val_batch_size: int = 64
    learning_rate: float = 2e-3
    num_epochs: int = 100
    device: typing.Optional[str] = 'cpu'


class DebuggingParams:
    batch_size: int = 64
    val_batch_size: int = 64
    learning_rate: float = 2e-3
    num_epochs: int = 3
    device: typing.Optional[str] = 'cpu'


class GNN(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(GNN, self).__init__()

        self.data_dir = data_dir or os.getcwd()  # pass this from now on

        self.node_feature_dim = config['node_feature_dim']
        self.edge_dim = config['edge_dim']
        self.hidden_size = config['embedding_dim']

        self.gat = GATConv(in_channels=(self.node_feature_dim, self.edge_dim), out_channels=self.hidden_size, edge_dim=3)

        self.final_lin = nn.Linear(self.hidden_size, 1)

    def forward(self, graphs: Data):
        """
        Produces a column vector of predictions, with each element in this vector a prediction
        for each marked graph in `graphs_in`.

        In the comments below N is the number of nodes in graph_in (across all graphs),
        d the feature dimension, and G is the number of individual molecular graphs.
        """

        # 1. Message passing and updating
        x, edge_attr, edge_index = graphs.x, graphs.edge_attr, graphs.edge_index
        x = x.float()
        edge_attr = edge_attr.float()
        graph_embedding = self.gat(x, edge_index=edge_index, edge_attr=edge_attr)
        graph_embedding = F.relu(graph_embedding)


        # 3. Final linear projection.
        final_prediction = self.final_lin(graph_embedding)  # [G, 1]
        return final_prediction

    def mse_loss(self, prediction, target):
        result = F.mse_loss(prediction, target)
        return result

    def training_step(self,train_batch, batch_idx):
        x, y = train_batch
        print(f'printing x:{x}')
        print(f'shape of y:{y.shape}') # TODO continue here, for some reason only 1 graph gets passed in but a vector of multiple y (target) values, is this correct???
        prediction = self.forward(x)
        print(f'printing prediction:{prediction}')
        loss = self.mse_loss(prediction, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        graphs, y = val_batch
        prediction = self.forward(graphs)
        loss = self.mse_loss(prediction, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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