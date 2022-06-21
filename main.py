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
def main():
    pretrain_epochs = 2
    finetune_epochs = 100

from datetime import datetime

def main():
    best_loss, best_config = meta_hpo_finetuning(pretrain_epochs = 50,
                                                 finetune_epochs = 30,
                                                 n_samples = 30,
                                                 train_size = 0.8)
    save_loss_and_config(best_loss, best_config)
    print('Completed a pretrain and finetuning HPO run!')

if __name__ == '__main__':
    main()
