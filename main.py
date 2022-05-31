import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from rdkit import Chem
from rdkit.Chem import PandasTools
from utils.mol2fingerprint import calc_fps
from utils.preprocessing_to_smiles import canonical_smiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from utils.mol2graph import Graphs
import torch
from ray import tune
from network import GNN, GNNDataModule, train_neural_network, plot_train_and_val_using_altair, collate_for_graphs, plot_train_and_val_using_mpl, TrainParams, DebuggingParams

from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


raytune_callback = TuneReportCallback(
    {
        'loss' : 'val_loss',
        'mean_accuracy' : 'val_accuracy'
    },
    on='validation_end')

def train_tune(config, max_epochs=10, gpus=0):
    gnn_config = {
        'node_feature_dimension' : config['node_feature_dimension'],
        'num_propagation_steps' : config['num_propagation_steps'],
        'embeddings_dimension' : config['embeddings_dimension']
    }# we only split the configs here because of things like batch size, which is passed to DataModule while the other hyperparameters
    # are passed to the GNN class.
    datamodule_config = {
        'collate_func' : config['collate_func'],
        'train_batch_size' : config['train_batch_size'],
        'val_batch_size' : config['val_batch_size'],
        'num_workers'   :   config['num_workers'],
        'data_dir'  :   config['data_dir']
    }

    model = GNN(gnn_config)
    datamodule = GNNDataModule(datamodule_config)

    trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator='gpu',
                            devices=1,
                            enable_progress_bar=True,
                            enable_checkpointing = True,
                            callbacks=[raytune_callback])
    trainer.fit(model, datamodule)
#    trainer.test(model, data_module) #loads the best checkpoint automatically


def main():

    torch.set_default_dtype(torch.float64)
    node_feature_dimension = len(Graphs.ATOM_FEATURIZER.indx2atm)

    config = {
        'node_feature_dimension' : len(Graphs.ATOM_FEATURIZER.indx2atm),
        'num_propagation_steps' : tune.randint(1, 10),
        'embeddings_dimension' : len(Graphs.ATOM_FEATURIZER.indx2atm),
        'collate_func' : collate_for_graphs,
        'train_batch_size' : tune.grid_search([4, 8, 16, 32, 64, 128]),
        'val_batch_size' : tune.grid_search([4, 8, 16, 32, 64, 128]),
        'num_workers'   :   0,
        'data_dir'  :   'C:\\Users\\bwvan\\PycharmProjects\\GenMol\\data\\'
    }
    tune.run(partial(train_tune, max_epochs=30, gpus=1),
             config=config,
             num_samples=3)

if __name__ == '__main__':
    main()