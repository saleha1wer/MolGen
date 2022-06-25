from ast import Raise
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
from torch.nn import functional as F, Linear, BatchNorm1d, ModuleList, ReLU, Sequential
from torch_geometric.nn.glob import GlobalAttention, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN, GAT, PNA

from utils.GINE import GINE


class GNN(pl.LightningModule):
    def __init__(self, config, data_dir=None,name='GNN'):
        super(GNN, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        self.input_heads = config['input_heads']
        self.learning_rate = config['lr']
        self.num_features = config['N']
        self.edge_dim = config['E']
        self.layer_type = config['layer_type']
        self.hidden_size = config['hidden']
        self.num_layers = config['n_layers']
        self.batch_size = config['batch_size']
        self.dropout_rate = config['dropout_rate']
        self.second_input = config['second_input']
        dim = self.hidden_size

        if self.layer_type == GAT:
            self.gnn = self.layer_type(self.num_features, dim, num_layers=self.num_layers, edge_dim=self.edge_dim,heads=8, norm=torch.nn.BatchNorm1d(dim))


        elif self.layer_type == GINE:
            self.gnn = self.layer_type(self.num_features,dim, num_layers=self.num_layers,edge_dim=self.edge_dim,norm=torch.nn.BatchNorm1d(dim))

        layers = self.gnn._modules['convs']

        if config['active_layer'] == 'first':
            self.last_layer = layers[0]
        elif config['active_layer'] == 'last':
            self.last_layer = layers[:-1]

        if config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif config['pool'] == 'GlobalAttention':
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size, 1))
        else:
            raise ValueError('pool type not supported')

        self.fc1 = Linear(dim, dim)

        self.fc2 = Linear(dim, 1)

        self.save_hyperparameters()
        self.emb_f = None

    def forward(self, graphs: Data):
        batch = graphs.batch
        if isinstance(self.gnn, GAT):
            x = graphs.x[:, :self.num_features].to(torch.float)
            edge_attr = graphs.edge_attr[:, :self.edge_dim].to(torch.float)
            edge_index = graphs.edge_index
            x = F.relu(self.gnn(x, edge_index, edge_attr))
        # elif isinstance(self.gnn, GIN):
        #     x = graphs.x[:, :self.num_features].to(torch.float)
        #     edge_index = graphs.edge_index
        #     x = F.relu(self.gnn(x, edge_index))
        elif isinstance(self.gnn, GINE):
            x = graphs.x[:, :self.num_features].to(torch.float)
            edge_attr = graphs.edge_attr[:, :self.edge_dim].to(torch.float)
            edge_index = graphs.edge_index
            x = F.relu(self.gnn(x, edge_index, edge_attr))
        self.emb_f = self.pool(x, batch)
        x = F.relu(self.fc1(self.emb_f))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.fc2(x)
        return x

    def mse_loss(self, prediction, target):
        # prediction = prediction.reshape(target.shape)
        result = F.mse_loss(prediction, target)
        return result

    def training_step(self, train_batch, batch_idx):
        prediction = self.forward(train_batch)
        loss = self.mse_loss(prediction, train_batch.y)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        prediction = self.forward(val_batch)
        loss = self.mse_loss(prediction, val_batch.y)
        self.log('val_loss', loss, batch_size=self.batch_size)
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        prediction = self.forward(test_batch)
        loss = self.mse_loss(prediction, test_batch.y)
        self.log('test_loss', loss, batch_size=self.batch_size)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_results = {'test_loss': avg_loss}
        return self.test_results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_bottleneck(self):
        return self.emb_f
