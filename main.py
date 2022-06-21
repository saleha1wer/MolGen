import os
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
from datetime import datetime
import ray


def main():
    ray.init(_memory = 3*1024*1024*1024)
    pretrain_epochs = 50
    finetune_epochs = 30
    adenosine_star = False
    NUM_NODE_FEATURES = 5
    NUM_EDGE_FEATURES = 1
    n_samples = 30 #hpo param
    max_t_per_trial = 2000 #hpo param
    batch_size = 64
    no_a2a = True #use a2a data or not in adenosine set
    no_a2a = '_no_a2a' if no_a2a else ''
    train_size = 0.8
    # for prot_target_encoding choose: None or 'one-hot-encoding'
    # if choosing one-hot-encoding change input_heads in gnn_config

    #
    ######################################################################################################################p
    # HPO on pretrain data (adenosine)
    # best_config = run_hpo(max_epochs=max_epochs, n_samples=n_samples, max_t_per_trial=max_t_per_trial, data_module=pre_data_module, gnn_config=gnn_config)
    # print('BEST CONFIG: ')
    # print(best_config)
    # Pretrain best config on pretrain data
    gnn_config = {
        'N': NUM_NODE_FEATURES,
        'E': NUM_EDGE_FEATURES,
        'lr': tune.loguniform(1e-4, 1e-1),  # learning rate
        'hidden': tune.choice([16, 32, 64, 128, 256, 512]),  # embedding/hidden dimensions
        # 'layer_type': tune.choice([GIN, GAT, GraphSAGE]),
        'layer_type': GIN,
        'n_layers': tune.choice([2, 3, 4, 5, 6, 7]),
        'pool': tune.choice(['mean', 'GlobalAttention']),
        'accelerator': 'cpu',
        'batch_size': batch_size,
        'input_heads': 1,
        'active_layer': tune.choice(['first', 'last']),
        'trade_off_backbone': tune.loguniform(1e-5, 10),
        'trade_off_head': tune.loguniform(1e-5, 1),
        'order': tune.choice([1, 2, 3, 4]), #is this a good interval?
        'patience': 10
        # 'batch_size': tune.choice([16,32,64,128])
    }
    pre_data_module, fine_data_module = create_pretraining_finetuning_DataModules(batch_size, no_a2a, train_size)

    best_configuration, best_loss = run_hpo_finetuning(pretrain_epochs, finetune_epochs, n_samples, max_t_per_trial, pre_data_module, fine_data_module, gnn_config)

    now_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    file = open('HPO_results.txt','a')
    message = f"\nTime of writing: {now_string}\nLoss achieved: {str(best_loss)} \nConfiguration found: {str(best_configuration)}"
    file.write(message)

if __name__ == '__main__':
    main()
