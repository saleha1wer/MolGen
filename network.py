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
from torch import nn, tensor,concat
from torch.nn import functional as F, Linear, BatchNorm1d, ModuleList, ReLU, Sequential
from torch_geometric.nn.glob import GlobalAttention, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN, PNA

from utils.GINE import GINE,GAT


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
            self.gnn = self.layer_type(self.num_features, dim, num_layers=self.num_layers, edge_dim=self.edge_dim,v2=config['v2'],heads=8, norm=torch.nn.BatchNorm1d(dim))
        elif self.layer_type == GINE:
            if self.edge_dim == 0:
                self.gnn = GIN(self.num_features,dim, num_layers=self.num_layers,norm=torch.nn.BatchNorm1d(dim))
            else:
                self.gnn = self.layer_type(self.num_features,dim, num_layers=self.num_layers, edge_dim=self.edge_dim,norm=torch.nn.BatchNorm1d(dim))
            # self.mlp = Sequential(Linear(self.edge_dim,self.num_features))
        else:
            raise ValueError('Unknown layer type: {}'.format(self.layer_type))

        layers = self.gnn._modules['convs']
        self.last_layer = layers[-1]

        if config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif config['pool'] == 'GlobalAttention':
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size, 1))
        else:
            raise ValueError('pool type not supported')

        self.fc1 = Linear(dim, dim)

        self.fc2 = Linear(dim, 1)


        if self.input_heads == 1:
            self.fc2 = Linear(dim, 1)

        elif self.input_heads == 2:
            self.second_input = config['second_input']
            if self.second_input == 'prot':
                self.fc_1 = Linear(4, dim)
                self.fc2 = Linear(dim*2, 1)

            elif self.second_input == 'xgb':
                self.fc_1 = Linear(1,dim)
                self.fc2 = Linear(int(dim/4), 1)
                self.fc_2 = Linear(2*dim, int(dim/4))
                self.fc_3 = Linear(int(dim/4), int(dim/4))
            elif self.second_input == 'fps':
                self.fc_1 = Linear(2067,dim)
                self.fc2 = Linear(int(dim/4), 1)
                self.fc_2 = Linear(2*dim, int(dim/4))
                self.fc_3 = Linear(int(dim/4), int(dim/4))


        self.save_hyperparameters()
        self.emb_f = None

    def forward(self, graphs: Data):
        batch = graphs.batch
        if self.layer_type == GAT:
            x = graphs.x[:, :self.num_features].to(torch.float)
            edge_attr = graphs.edge_attr[:, :self.edge_dim].to(torch.float)
            edge_index = graphs.edge_index
            x = F.relu(self.gnn(x, edge_index, edge_attr))
        elif self.layer_type == GINE:
            x = graphs.x[:, :self.num_features].to(torch.float)
            edge_attr = graphs.edge_attr[:, :self.edge_dim].to(torch.float)
            edge_index = graphs.edge_index
            # edge_attr = self.mlp(edge_attr)
            if self.edge_dim == 0:
                x = F.relu(self.gnn(x, edge_index))
            else:
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
