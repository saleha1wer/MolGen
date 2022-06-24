import os
from tkinter import N
from flask import Config
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from network import GNN
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray import tune
from torch_geometric.data import Data
import multiprocessing as mp   
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
from utils.encode_ligand import calc_fps
from rdkit import Chem
from xgboost import XGBRegressor

def temp_func(n,e,config,data_module,test_loader):
    config['N'] = n
    config['E'] = e
    results = []
    for i in range(3):
        model = GNN(config)
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=100)
        trainer.fit(model, data_module)    
        res = trainer.test(model,test_loader)
        res = res[0]['test_loss']
        results.append(res)
    score = np.mean(results)
    std = np.std(results)
    return [n,e,score,std]

def main():
    batch_size = 64
    
    parameters = {'N': [1, 2,4, 5, 7, 8, 9], 'E': [0, 1, 3]}
    
    config = {'N': None, 'E': None, 'lr': 0.0003, 'hidden': 256,
              'layer_type': GIN, 'n_layers': 4, 'pool': 'mean', 'accelerator': 'cpu',
              'batch_size': 64, 'input_heads': 1, 'active_layer': 'first', 'trade_off_backbone': 1,
              'trade_off_head':0.0005, 'order': 1, 'patience': 10, 'dropout_rate':0.5, 'second_input': None}
    
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
    test_loader = data_module.test_dataloader()

        
    n_cores = int(mp.cpu_count())
    pool = mp.Pool(processes=int(n_cores))
    results = pool.starmap(temp_func, [(n_node,0,config,data_module,test_loader) for n_node in parameters['N']])
    # results = pool.starmap(temp_func, [(n_node,n_edge,config,data_module,test_loader) for n_node in parameters['N'] for n_edge in parameters['E']])
    print(results)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()