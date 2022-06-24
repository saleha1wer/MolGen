import os
from flask import Config
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from network import GNN
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray.tune.utils import wait_for_gpu
from hpo import run_hpo_basic, run_hpo_finetuning, meta_hpo_finetuning, save_loss_and_config, calculate_test_loss, meta_hpo_basic
from finetune import finetune
import torch
from ray import tune


def main(hpo_ft):
    if hpo_ft:
        finetune_epochs = 50
        patience = 15
        n_samples = 50
        space = {'order':1, 'trade_off_head': 0.00005, 'trade_off_backbone':1}

        source_model = ... # load pretrained model
        best_val_loss, best_configuration = meta_hpo_finetuning(finetune_epochs, patience, n_samples, 0.9, space)
    else:
        pretrain_epochs = 50
        train_size = 0.9
        no_a2a = True
        batch_size = 64
        n_inputs = 1
        second_input = 'fps'
        space = {'N': 9, 'E': 1, 'lr': tune.loguniform(1e-5, 1e-2), 'hidden': tune.choice([64, 128, 256, 512, 1024]),
                'layer_type': GAT, 'n_layers': tune.choice([2,4,6,8]), 'pool': tune.choice(['mean', 'GlobalAttention']),
                'dropout_rate' : tune.choice([0,0.1,0.3,0.5]), 'accelerator': 'cpu', 'batch_size': 64,
                'input_heads': 1, 'active_layer': 'first', 'second_input': None}
        
        best_val_loss,best_test_loss, best_config = meta_hpo_basic(pretrain_epochs,
                                                    n_samples = 50,
                                                    train_size = train_size,
                                                    space = space,
                                                    report_test_loss = True)

        save_loss_and_config(best_val_loss,best_val_loss, best_config)
        print('Completed a basic HPO run!')


if __name__ == '__main__':
    main(hpo_ft=True)
