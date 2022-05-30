import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit.Chem import PandasTools
from utils.mol2fingerprint import calc_fps
from utils.preprocessing_to_smiles import canonical_smiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from utils.mol2graph import Graphs
from network import GNN, GNNDataModule, train_neural_network, plot_train_and_val_using_altair, collate_for_graphs, plot_train_and_val_using_mpl, TrainParams, DebuggingParams
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
def main():

    torch.set_default_dtype(torch.float64)
    node_feature_dimension = len(Graphs.ATOM_FEATURIZER.indx2atm)
    num_propagation_steps = 4 # default value?

    batch_size = 64

    gnn_config = {
        'node_feature_dimension' : node_feature_dimension,
        'num_propagation_steps' : num_propagation_steps
    }
    datamodule_config = {
        'collate_func' : collate_for_graphs,
        'train_batch_size' : batch_size,
        'val_batch_size' : batch_size,
        'num_workers'   :   2
    }

    model = GNN(gnn_config)
    data_module = GNNDataModule(datamodule_config)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)

    trainer.fit(model, data_module)

    # Here we put the testing code




if __name__ == '__main__':
    main()