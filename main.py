import os
from flask import Config
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from network import GNN
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray.tune.utils import wait_for_gpu
from hpo import run_hpo_basic, run_hpo_finetuning, meta_hpo_finetuning, save_loss_and_config, calculate_test_loss
from finetune import finetune
import torch
from ray import tune


def main():
    pretrain_epochs = 2
    finetune_epochs = 2
    best_val_loss, best_test_loss, best_config = meta_hpo_finetuning(pretrain_epochs = pretrain_epochs,
                                                 finetune_epochs = finetune_epochs,
                                                 n_samples = 1,
                                                 train_size = 0.9,
                                                 report_test_loss = True)

    save_loss_and_config(best_val_loss, best_test_loss, best_config)
    print('Completed a pretrain and finetuning HPO run!')
    


if __name__ == '__main__':
    main()
