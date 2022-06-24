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
from utils.GINE_network import GINE

class GNN_GAT(pl.LightningModule):
    def __init__(self, config, data_dir=None, name='GAT'):
        super(GNN_GAT, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        self.input_heads = config['input_heads']
        self.learning_rate = config['lr']
        self.num_features = config['N']
        self.edge_dim = config['E']
        self.hidden_size = config['hidden']
        self.layer_type = config['layer_type']
        self.num_layers = config['n_layers']
        self.batch_size = config['batch_size']
        self.dropout_rate = config['dropout_rate']
        self.second_input= config['second_input']
        dim = self.hidden_size

        self.gnn = self.layer_type(self.num_features, dim, num_layers=self.num_layers, edge_dim=self.edge_dim,
                                   heads=8, norm=torch.nn.BatchNorm1d(dim))

        self.node_embedding = Linear(in_features=self.num_features, out_features=self.hidden_size)
        self.edge_embedding = Linear(in_features=self.edge_dim, out_features=self.hidden_size)
        
        if config['active_layer'] == 'first':
            self.last_layer = self.gnn._modules['convs'][0]
        elif config['active_layer'] == 'last':
            self.last_layer = self.gnn._modules['convs'][self.num_layers-1]

        if config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif config['pool'] == 'GlobalAttention':
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size, 1))
        else:
            raise ValueError('pool type not supported')

        self.fc1 = Linear(dim, dim)

        if self.input_heads == 1:
            self.fc2 = Linear(dim, 1)

        elif self.input_heads == 2:
            self.second_input = config['second_input']
            if self.second_input == 'prot':
                self.fc_1 = Linear(4, dim)
            elif self.second_input == 'xgb':
                self.fc_1 = Linear(1,dim)
            elif self.second_input == 'fps':
                self.fc_1 = Linear(2067,dim)
            self.fc_2 = Linear(2*dim, int(dim/4))
            self.fc_3 = Linear(int(dim/4), int(dim/4))
            self.fc2 = Linear(int(dim/4), 1)

        self.save_hyperparameters()
        self.emb_f = None

    def forward(self, graphs: Data):
        # x = self.node_embedding(graphs.x[:, :self.num_features].to(torch.float))
        # edge_attr = self.edge_embedding(graphs.edge_attr[:, :self.edge_dim].to(torch.float))
        x = graphs.x[:, :self.num_features].to(torch.float)
        edge_attr = graphs.edge_attr[:, :self.edge_dim].to(torch.float)
        edge_index = graphs.edge_index
        batch = graphs.batch
        x = F.relu(self.gnn(x, edge_index, edge_attr))
        self.emb_f = self.pool(x, batch)
        x = F.relu(self.fc1(self.emb_f))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.input_heads == 2:
            if self.second_input == 'prot':
                p = self.fc_1(graphs.p)
                x = torch.concat((x, p), dim=-1)  # on PyTorch 1.9 use torch.ca

            elif self.second_input == 'xgb':
                p = graphs.xgb_pred
                p = p.reshape(p.shape[0],1)
                p = self.fc_1(p)
                p = F.dropout(p, p=self.dropout_rate, training=self.training)
                x = torch.cat((x, p), dim=1)
                x = self.fc_2(x)
                x = self.fc_3(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            elif self.second_input == 'fps':
                p = graphs.fps.float()
                p = self.fc_1(p)
                x = torch.cat((x, p), dim=1)
                x = self.fc_2(x)
                x = self.fc_3(x)
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


class GNN_GINE(pl.LightningModule):
    def __init__(self, config, data_dir=None, name='GAT'):
        super(GNN_GINE, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        self.input_heads = config['input_heads']
        self.learning_rate = config['lr']
        self.num_features = config['N']
        self.edge_dim = config['E']
        self.hidden_size = config['hidden']
        self.layer_type = config['layer_type']
        self.num_layers = config['n_layers']
        self.batch_size = config['batch_size']
        self.dropout_rate = config['dropout_rate']
        self.second_input= config['second_input']
        dim = self.hidden_size
        self.layer_type = GINE
        if self.edge_dim == 0:
            self.gnn = self.layer_type(self.num_features, dim, num_layers=self.num_layers,
                                   norm=torch.nn.BatchNorm1d(dim))
        if self.edge_dim != 0:
            self.gnn = self.layer_type(self.num_features, dim, num_layers=self.num_layers,
                                       norm=torch.nn.BatchNorm1d(dim))

        # self.node_embedding = Linear(in_features=self.num_features, out_features=self.hidden_size)

        self.edge_embedding = Linear(in_features=self.edge_dim, out_features=self.num_features)

        if config['active_layer'] == 'first':
            self.last_layer = self.gnn._modules['convs'][0]
        elif config['active_layer'] == 'last':
            self.last_layer = self.gnn._modules['convs'][self.num_layers-1]

        if config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif config['pool'] == 'GlobalAttention':
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size, 1))
        else:
            raise ValueError('pool type not supported')

        self.fc1 = Linear(dim, dim)

        if self.input_heads == 1:
            self.fc2 = Linear(dim, 1)

        elif self.input_heads == 2:
            self.second_input = config['second_input']
            if self.second_input == 'prot':
                self.fc_1 = Linear(4, dim)

            elif self.second_input == 'xgb':
                self.fc_1 = Linear(1,dim)
            elif self.second_input == 'fps':
                self.fc_1 = Linear(2067,dim)

            self.fc_2 = Linear(2*dim, int(dim/4))
            self.fc_3 = Linear(int(dim/4), int(dim/4))

            self.fc2 = Linear(int(dim/4), 1)

        self.save_hyperparameters()
        self.emb_f = None

    def forward(self, graphs: Data):
        x = graphs.x[:, :self.num_features].to(torch.float)
        edge_index = graphs.edge_index
        batch = graphs.batch
        edge_attr = graphs.edge_attr[:, :self.edge_dim].to(torch.float)
        if self.edge_dim != 0:
            edge_attr = self.edge_embedding(edge_attr)
        x = F.relu(self.gnn(x, edge_index, edge_attr))
        self.emb_f = self.pool(x, batch)
        x = F.relu(self.fc1(self.emb_f))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.input_heads == 2:
            if self.second_input == 'prot':
                p = self.fc_1(graphs.p)
                x = torch.concat((x, p), dim=-1)  # on PyTorch 1.9 use torch.ca

            elif self.second_input == 'xgb':
                p = graphs.xgb_pred
                p = p.reshape(p.shape[0],1)
                p = self.fc_1(p)
                p = F.dropout(p, p=self.dropout_rate, training=self.training)
                x = torch.cat((x, p), dim=1)
                x = self.fc_2(x)
                x = self.fc_3(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            elif self.second_input == 'fps':
                p = graphs.fps.float()
                p = self.fc_1(p)
                x = torch.cat((x, p), dim=1)
                x = self.fc_2(x)
                x = self.fc_3(x)
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