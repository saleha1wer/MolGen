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
from torch.nn import functional as F
from torch.utils import data
from torch_geometric.nn import TransformerConv, TopKPooling, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.models import GAT
from torch_geometric.data import Data

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling, GIN
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool

from torch_geometric.nn.conv import GINConv, PANConv,GATv2Conv
from torch_geometric.nn.models import GIN, GAT, PNA
from torch.nn import ReLU, Sequential


class GNN(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(GNN, self).__init__()

        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        #
        self.learning_rate = config['lr']
        num_features = config['N']
        self.edge_dim = config['E']
        self.hidden_size = config['hidden']
        self.layer_type = config['layer_type']
        self.num_layers = config['n_layers']
        dim = self.hidden_size
        self.gnn = self.layer_type(num_features, dim, self.num_layers,norm=torch.nn.BatchNorm1d(dim))
        # nn1 = nn.Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        # self.conv1 = GINConv(nn1)
        # self.bn1 = torch.nn.BatchNorm1d(dim)

        # nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv2 = GINConv(nn2)
        # self.bn2 = torch.nn.BatchNorm1d(dim)

        # nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv3 = GINConv(nn3)
        # self.bn3 = torch.nn.BatchNorm1d(dim)

        # nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv4 = GINConv(nn4)
        # self.bn4 = torch.nn.BatchNorm1d(dim)

        # nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv5 = GINConv(nn5)
        # self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 1)
        self.save_hyperparameters()

    def forward(self, graphs):
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch
        x = F.relu(self.gnn(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

        # self.data_dir = data_dir or os.getcwd()  # pass this from now on
        #
        # self.learning_rate = config['learning_rate']
        # self.node_feature_dim = config['node_feature_dimension']
        # self.edge_dim = config['edge_feature_dimension']
        # self.hidden_size = config['embedding_dimension']
        # self.propagation_steps = config['num_propagation_steps']
        #
        # embedding_size = self.hidden_size
        # n_heads = 8
        # self.n_layers = self.propagation_steps
        # dropout_rate = 0.1
        # top_k_ratio = 0.5
        # self.top_k_every_n = 1
        # dense_neurons = 256
        # edge_dim = self.edge_dim
        #
        # self.conv_layers = ModuleList([])
        # self.transf_layers = ModuleList([])
        # self.pooling_layers = ModuleList([])
        # self.bn_layers = ModuleList([])
        #
        # # Transformation layer
        # self.conv1 = TransformerConv(self.node_feature_dim,
        #                              embedding_size,
        #                              heads=n_heads,
        #                              dropout=dropout_rate,
        #                              edge_dim=edge_dim,
        #                              beta=True)
        #
        # self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        # self.bn1 = BatchNorm1d(embedding_size)
        #
        # # Other layers
        # for i in range(self.n_layers):
        #     self.conv_layers.append(TransformerConv(embedding_size,
        #                                             embedding_size,
        #                                             heads=n_heads,
        #                                             dropout=dropout_rate,
        #                                             edge_dim=edge_dim,
        #                                             beta=True))
        #
        #     self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
        #     self.bn_layers.append(BatchNorm1d(embedding_size))
        #     if i % self.top_k_every_n == 0:
        #         self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))
        #
        # # Linear layers
        # self.linear1 = Linear(embedding_size * 2, dense_neurons)
        # self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        # self.linear3 = Linear(int(dense_neurons / 2), 1)
        #
        # # self.gat = GAT(self.node_feature_dim, hidden_channels=64, num_layers=4, out_channels=self.hidden_size,
        # #                dropout=0.1, edge_dim=self.edge_dim)
        #
        # self.final_lin = nn.Linear(self.hidden_size, 1)

    def mse_loss(self, prediction, target):
        prediction = prediction.reshape(target.shape)
        result = F.mse_loss(prediction, target)
        return result

    def training_step(self, train_batch, batch_idx):
        prediction = self.forward(train_batch)
        loss = self.mse_loss(prediction, train_batch.y)
        self.log('train_loss', loss, batch_size=64)
        return loss

    def validation_step(self, val_batch, batch_idx):
        prediction = self.forward(val_batch)
        loss = self.mse_loss(prediction, val_batch.y)
        self.log('val_loss', loss, batch_size=64)
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        prediction = self.forward(test_batch)
        loss = self.mse_loss(prediction, test_batch.y)
        self.log('test_loss', loss, batch_size=64)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_results = {'test_loss': avg_loss}
        return self.test_results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
