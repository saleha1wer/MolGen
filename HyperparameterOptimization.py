import pandas as pd
import numpy as np
import typing
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from filelock import FileLock
import torch
import torch.cuda
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
import os
from ray import tune
from rdkit import Chem
from rdkit.Chem import PandasTools
from utils.mol2fingerprint import calc_fps
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from utils.mol2graph import Graphs, GraphRegressionDataset
from network import GNN, train_neural_network, plot_train_and_val_using_altair, collate_for_graphs, plot_train_and_val_using_mpl

from ignite.contrib.handlers import ProgressBar

def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)

def create_GNN(hyperparameters): # for now has dummy parameters
    network = GNN(len(Graphs.ATOM_FEATURIZER.indx2atm))
    return network

#def train_gnn_tune(config, num_epochs=10, num_gpus=0, data_dir=)


# def GNN_hyperparameter_optimization(train_dataset : np.ndarray,
#                                     val_dataset : np.ndarray,
#                                     collate_func: typing.Optional[typing.Callable]=None):
#     print(f"Train dataset is of size {len(train_dataset)} and valid of size {len(val_dataset)}")
#
#     train_dataset = GraphRegressionDataset.create_from_df(train_dataset)
#     val_dataset = GraphRegressionDataset.create_from_df(val_dataset)
#
#
#     def single_run(config):
#         train_dataloader = data.DataLoader(train_dataset, config['batch_size'], shuffle=True,
#                                            collate_fn=collate_func, num_workers=1)
#         val_dataloader = data.DataLoader(val_dataset, config['batch_size'], shuffle=False, collate_fn=collate_func,
#                                            num_workers=1)
#         # neural network
#         model = create_GNN(config['hyperparameters'])
#
#         # Optimizer
#         optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
#
#         # for now let's ignore the device parameter in the params given to this function and use this instead
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # Work out what device we're going to run on (ie CPU or GPU)
#         print('Starting training of NN on device: {}'.format(device))
#
#         def prepare_batch(batch, device, non_blocking):
#             x, y = batch
#             return x.to(device), y.to(device)
#
#         trainer = create_supervised_trainer(model, optimizer, F.mse_loss, device=device,
#                                             prepare_batch=prepare_batch)
#         evaluator = create_supervised_evaluator(model,
#                                                 metrics={'loss': Loss(F.mse_loss)},
#                                                 device=device, prepare_batch=prepare_batch)
#
#         # progress bar, superfluous? Remove if it gives errors in combination with the Ray tune framework
#         RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
#
#         pbar = ProgressBar(persist=True)
#         pbar.attach(trainer, metric_names='all')
#
#         # remove below
#         for i in range(config['num_epochs']):
#             print(f'Epoch {i + 1}\n------------------')
#             train_loop(train_dataloader, model, loss_fn, optimizer)
#             test_loop(test_dataloader, model, loss_fn)
#         print('Done training')
#
#         output = train_neural_network(train_dataset=graphs_train, val_dataset=graphs_val, params=params,
#                                       neural_network=network, collate_func=collate_for_graphs)  # ...ETC)
#
#         # plot = plot_train_and_val_using_altair(out['train_loss_list'], out['val_lost_list'])
#         # save(plot, 'chart_lr=2e-3.png')  # .pdf doesn't work?
#
#         plot_train_and_val_using_mpl(output['train_loss_list'], output['val_lost_list'])
#         plt.savefig('char_lr=2e-3.pdf')
#
#         plot_train_and_val_using_mpl(output['train_loss_list'], output['val_lost_list'], name=name_of_run, save=True)
#
#     single_run(config)