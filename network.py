# THIS WORK IS FROM JOHN BRADSHAW - https://github.com/john-bradshaw/ml-in-bioinformatics-summer-school-2020
## THIS SCRIPT INCLUDES HOW TO TRAIN A GNN AND AN EXAMPLE GNN (AND PLOTTING)

import os
import time
import typing
import collections
import itertools
from utils.mol2fingerprint import calc_fps
from utils.mol2graph import Graphs, GraphRegressionDataset
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback


class GNN(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(GNN, self).__init__()

        self.learning_rate = config['learning_rate']
        self.node_feature_dimension = config['node_feature_dimension']
        self.embeddings_dimension = config[
            'embeddings_dimension']  # used to be node_feature_dimension, vary this for hyperparam. opt.
        self.num_propagation_steps = config['num_propagation_steps']
        # called T above.

        # Our sub modules:
        self.message_projection = nn.Linear(self.node_feature_dimension, self.node_feature_dimension, bias=False)
        self.update_gru = nn.GRUCell(input_size=self.node_feature_dimension,
                                     hidden_size=self.embeddings_dimension, bias=True)
        self.attn_net = nn.Linear(self.embeddings_dimension, 1)
        self.proj_net = nn.Linear(self.embeddings_dimension, self.embeddings_dimension)
        self.final_lin = nn.Linear(self.embeddings_dimension, 1)
        self.double()

    def forward(self, graphs_in: Graphs):
        """
        Produces a column vector of predictions, with each element in this vector a prediction
        for each marked graph in `graphs_in`.

        In the comments below N is the number of nodes in graph_in (across all graphs),
        d the feature dimension, and G is the number of individual molecular graphs.
        """
        # 1. Message passing and updating
        m = graphs_in.node_features  # shape: [N, d]
        for t in range(self.num_propagation_steps):
            projs = self.message_projection(m)  # [N, d]

            # Update the node embeddings (eqn 1 above)
            # 1a. compute the sum for each node
            msgs = torch.zeros_like(m)  # [N, d]
            msgs.index_add_(0, graphs_in.edge_list[:, 0], projs.index_select(0, graphs_in.edge_list[:, 1]))

            # 1b. update the embeddings via GRU cell
            m = self.update_gru(msgs, m)  # [N, d]

        # 2. Aggregation (eqn 2 above)
        # a compute weighted embeddings
        attn_coeffs = torch.sigmoid(self.attn_net(m))  # [N, 1]
        proj_embeddings = self.proj_net(m)  # [N, d']
        weighted_embeddings = attn_coeffs * proj_embeddings

        # perform the sum
        graph_embedding = torch.zeros(graphs_in.num_graphs, weighted_embeddings.shape[1],
                                      device=m.device, dtype=m.dtype)
        graph_embedding.index_add_(0, graphs_in.node_to_graph_id, weighted_embeddings)  # [G, d']

        # 3. Final linear projection.
        final_prediction = self.final_lin(graph_embedding)  # [G, 1]
        return final_prediction

    def mse_loss(self, prediction, target):
        result = F.mse_loss(prediction, target)
        return result

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        prediction = self.forward(x)
        loss = self.mse_loss(prediction, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        prediction = self.forward(x)
        loss = self.mse_loss(prediction, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        prediction = self.forward(x)
        loss = self.mse_loss(prediction, y)
        self.log('test_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# Here's the dataloader class (as in https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)
class GNNDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        if 'data_dir' in config:
            self.data_dir = config['data_dir']
        else:
            # DOESN'T WORK WITH RAYTUNE, raytune executes in a different folder than where the python files are so explicitly pass the file location every time
            self.data_dir = os.getcwd()
        self.prepare_data_per_node = True
        self.val_batch_size = config['val_batch_size']
        self.train_batch_size = config['train_batch_size']
        self.collate_func = config['collate_func']
        self.num_workers = config['num_workers']

    def setup(self, stage):  # don't know what the stage param is used for
        # for now the data paths are hardcoded
        print('Attempting to read directory:\n' 'human_adenosine_ligands')
        df = pd.read_csv(self.data_dir + 'human_adenosine_ligands')

        ####### !!!!!!!!!!!!!!!
        # df = df.head(500)  # for debugging
        print('Finished preprocessing Smiles')

        PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule', includeFingerprints=False)
        print('Processed Smiles to Mol object')

        target_values = df['pchembl_value_Mean'].to_numpy()

        fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEx method)
        print('Finished calculating fingerprints')

        graphs = np.array([[]])  # TUTORIAL METHOD (JOHN BRADSHAW)
        for mol in df['Molecule']:
            graphs = np.append(graphs, Graphs.from_mol(mol))
        print('Finished making graphs')

        y_train, y_test, fps_train, fps_test, graphs_train, graphs_test = train_test_split(target_values, fps, graphs,
                                                                                           test_size=0.2)
        graphs_train = graphs_train.reshape(graphs_train.shape[0], 1)
        y_train = y_train.reshape(y_train.shape[0], 1)
        graphs_test = graphs_test.reshape(graphs_test.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)
        self.train_data = GraphDataSet(data=graphs_train, targets=y_train)
        self.test_data = GraphDataSet(data=graphs_test, targets=y_test)
        print(f'loaded train_data of len:{len(self.train_data)}')
        print(f'loaded test_data of len:{len(self.test_data)}')

    def train_dataloader(self):
        train_dataloader = data.DataLoader(self.train_data,
                                           self.train_batch_size,
                                           shuffle=True,
                                           collate_fn=self.collate_func,
                                           num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = data.DataLoader(self.test_data,
                                         self.val_batch_size,
                                         shuffle=False,
                                         collate_fn=self.collate_func,
                                         num_workers=self.num_workers)
        return val_dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class GraphDataSet(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.transform = None  # can maybe be removed for us

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def collate_for_graphs(batch):
    """
    This is a custom collate function for use minibatches of graphs along with their regression value.
    It ensures that we concatenate graphs correctly.

    Look at ss_utils to see how this gets used.
    """
    # Split up the graphs and the y values

    zipped = zip(*batch)
    list_of_graphs, list_of_targets = zipped

    graphs = []

    for idx, item in enumerate(list_of_graphs):
        graphs.append(item[0])

    # The graphs need to be concatenated (i.e. collated) using the function you wrote
    graphs = Graphs.concatenate(graphs)

    # The y values can use the default collate function as before.
    targets = torch.utils.data.dataloader.default_collate(list_of_targets)

    return graphs, targets
