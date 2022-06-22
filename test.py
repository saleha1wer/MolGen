import os
from flask import Config
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from network import GNN
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray.tune.utils import wait_for_gpu
from hpo import run_hpo_basic, run_hpo_finetuning
from finetune import finetune
import torch
from ray import tune
from torch_geometric.data import Data

import pytorch_lightning as pl
import torch_geometric.nn.models.attentive_fp
from dataclasses import dataclass
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F, Linear, BatchNorm1d, ModuleList, ReLU, Sequential
from torch_geometric.nn.glob import GlobalAttention, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN, GAT, PNA, AttentiveFP
from utils.mol2fingerprint import calc_fps
from rdkit import Chem
from xgboost import XGBRegressor

def main():
    
    batch_size = 128
    
    parameters = {'N': [1, 4, 5, 7, 8, 9], 'E': [0, 1, 3]}
    
    config = {'N': 5, 'E': 3, 'lr': 0.00016542323876234363, 'hidden': 256,
              'layer_type': GATConv, 'n_layers': 6, 'pool': 'mean', 'accelerator': 'cpu',
              'batch_size': 64, 'input_heads': 1, 'active_layer': 'first', 'trade_off_backbone': 8.141935107421304e-05,
              'trade_off_head': 0.12425374868175541, 'order': 1, 'patience': 10}
    
    datamodule_config = {
        'batch_size': batch_size,
        'num_workers': 0
    }

    dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',
                                prot_target_encoding=None)

    train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.8, random_state=42)
    data_train = dataset[train_indices.tolist()]
    data_test = dataset[test_indices.tolist()]

    data_module = GNNDataModule(datamodule_config, data_train, data_test)


    for n_node in parameters['N']:
        for n_edge in parameters['E']:
            config['N'] = n_node
            config['E'] = n_edge
            model = GNN_edge(config)
            trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=200)
            trainer.fit(model, data_module)    


class GNN_edge(pl.LightningModule):
    def __init__(self, config, data_dir=None, name='GNN'):
        super(GNN_edge, self).__init__()
        self.name = name
        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        self.input_heads = config['input_heads']
        self.learning_rate = config['lr']
        self.num_features = config['N']
        self.edge_dim = config['E']
        self.hidden_size = config['hidden']
        self.layer_type = GATConv
        self.num_layers = config['n_layers']
        self.batch_size = config['batch_size']
        self.dim = self.hidden_size

        # GIN and GraphSAGE do not include edge attr
        # self.gnn = self.layer_type(num_features,dim, num_layers=self.num_layers, out_channels=1,
        #                            edge_dim=self.edge_dim, num_timesteps=4)
        self.gnn = self.layer_type(self.num_features, self.dim, edge_dim=self.edge_dim,
                                 dropout=0.5, heads=8)
        # if config['active_layer'] == 'first':
        #     self.last_layer = self.gnn._modules['convs'][0]
        # elif config['active_layer'] == 'last':
        #     self.last_layer = self.gnn._modules['convs'][self.num_layers-1]
        #
        if config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif config['pool'] == 'GlobalAttention':
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size*8, 1))
        else:
            raise ValueError('pool type not supported')

        self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.hidden_size*8, 1))

        self.fc1 = Linear(8*self.dim, 8*self.dim)

        if self.input_heads == 1:
            self.fc2 = Linear(self.dim*8, 1)

        # elif self.input_heads == 2:
        #     self.second_input = config['second_input']
        #     if self.second_input == 'prot':
        #         self.fc_1 = Linear(4, dim)
        #
        #     elif self.second_input == 'xgb':
        #         self.fc_1 = Linear(self.batch_size,dim)
        #
        #         # self.fc_3 = Linear(self.batch_size +1,self.batch_size)
        #     self.fc2 = Linear(2 * dim, 1)

        self.save_hyperparameters()
        self.emb_f = None

    def forward(self, graphs: Data):
        x, edge_index, batch, edge_attr = graphs.x[:,:self.num_features], graphs.edge_index, graphs.batch, \
                                          graphs.edge_attr[:,:self.edge_dim]
        
        print(x.shape, edge_index.shape, batch.shape, edge_attr.shape)
        x = F.relu(self.gnn(x, edge_index, edge_attr))
        self.emb_f = self.pool(x, batch)
        x = F.relu(self.fc1(self.emb_f))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        print('out shape', x.shape)
        return x

    def mse_loss(self, prediction, target):
        # prediction = prediction.reshape(target.shape)c
        result = F.mse_loss(prediction, target)
        return result

    def training_step(self, train_batch, batch_idx):
        prediction = self.forward(train_batch)
        loss = self.mse_loss(prediction, train_batch.y)
        self.log('train_loss', loss, self.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        prediction = self.forward(val_batch)
        loss = self.mse_loss(prediction, val_batch.y)
        self.log('val_loss', loss, batch_size=self.batch_size)
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        prediction = self.forward(test_batch)
        loss = self.mse_loss(prediction, test_batch.y)
        self.log('test_loss', loss, self.batch_size)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_results = {'test_loss': avg_loss}
        return self.test_results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_bottleneck(self):
        return self.emb_f

if __name__ == '__main__':
    main()


