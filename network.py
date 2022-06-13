# THIS WORK IS FROM JOHN BRADSHAW - https://github.com/john-bradshaw/ml-in-bioinformatics-summer-school-2020
## THIS SCRIPT INCLUDES HOW TO TRAIN A GNN AND AN EXAMPLE GNN (AND PLOTTING)

import os
# import time
# import typing
# import collections
# import itertools
from utils.mol2fingerprint import calc_fps
from utils.mol2graph import Graphs, GraphRegressionDataset
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import altair as alt
#from filelock import FileLock

#from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools

import torch
import torch
import os
import typing
import numpy as np
import time
import pandas as pd
import pytorch_lightning as pl
from dataclasses import dataclass

from matplotlib import pyplot as plt
from torch import nn
#from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch_geometric.nn import TransformerConv, TopKPooling, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.models import GAT
from torch_geometric.data import Data

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool

from torch_geometric.nn.conv import GINConv
from torch.nn import ReLU, Sequential

from torch.utils.data import DataLoader, random_split, Dataset
#from torchvision import transforms
import pytorch_lightning as pl

# from pytorch_lightning.loggers import TensorBoardLogger
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
# from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
#     TuneReportCheckpointCallback


class GNN(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super(GNN, self).__init__()

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
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        #        self.betas = config['betas']
        self.weight_decay = config['weight_decay']
        self.double()
        self.save_hyperparameters()

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
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_results = {'test_loss': avg_loss}
        return self.test_results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

class GNNJurren(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(GNN, self).__init__()

        self.data_dir = data_dir or os.getcwd()  # pass this from now on
        #
        self.learning_rate = config['learning_rate']
        num_features = config['node_feature_dimension']
        self.edge_dim = config['edge_feature_dimension']
        self.hidden_size = config['embedding_dimension']
        self.propagation_steps = config['num_propagation_steps']

        dim = self.hidden_size

        nn1 = nn.Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        """
        In the comments below N is the number of nodes in graph_in (across all graphs),
        d the feature dimension, and G is the number of individual molecular graphs.
        """
        # 1. Message passing and updating
        m = graphs_in.node_features  # shape: [N, d]
        for t in range(self.num_propagation_steps):
            projs = self.message_projection(m)  # [N, d]
        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 1)

    def forward(self, graphs):
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

        # self.data_dir = data_dir or os.getcwd()  # pass this from now on
        #
        # self.learning_rate = config['learning_rate']
        # self.node_feature_dim = config['node_feature_dimension']
        # self.edge_dim = config['edge_feature_dimension']
        # self.hidden_size = config['embedding_dimension']
        # self.propagation_steps = config['num_propagation_steps']
        #
        # embedding_size = self.hidden_size
        # n_heads = 8
        # self.n_layers = self.propagation_steps
        # dropout_rate = 0.1
        # top_k_ratio = 0.5
        # self.top_k_every_n = 1
        # dense_neurons = 256
        # edge_dim = self.edge_dim
        #
        # self.conv_layers = ModuleList([])
        # self.transf_layers = ModuleList([])
        # self.pooling_layers = ModuleList([])
        # self.bn_layers = ModuleList([])
        #
        # # Transformation layer
        # self.conv1 = TransformerConv(self.node_feature_dim,
        #                              embedding_size,
        #                              heads=n_heads,
        #                              dropout=dropout_rate,
        #                              edge_dim=edge_dim,
        #                              beta=True)
        #
        # self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        # self.bn1 = BatchNorm1d(embedding_size)
        #
        # # Other layers
        # for i in range(self.n_layers):
        #     self.conv_layers.append(TransformerConv(embedding_size,
        #                                             embedding_size,
        #                                             heads=n_heads,
        #                                             dropout=dropout_rate,
        #                                             edge_dim=edge_dim,
        #                                             beta=True))
        #
        #     self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
        #     self.bn_layers.append(BatchNorm1d(embedding_size))
        #     if i % self.top_k_every_n == 0:
        #         self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))
        #
        # # Linear layers
        # self.linear1 = Linear(embedding_size * 2, dense_neurons)
        # self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        # self.linear3 = Linear(int(dense_neurons / 2), 1)
        #
        # # self.gat = GAT(self.node_feature_dim, hidden_channels=64, num_layers=4, out_channels=self.hidden_size,
        # #                dropout=0.1, edge_dim=self.edge_dim)
        #
        # self.final_lin = nn.Linear(self.hidden_size, 1)

    # def forward(self, graphs):
    #     # Initial transformation
    #     x, edge_attr, edge_index, batch_index = graphs.x, graphs.edge_attr, graphs.edge_index, graphs.batch
    #     x = self.conv1(x, edge_index, edge_attr)
    #     x = torch.relu(self.transf1(x))
    #     x = self.bn1(x)
    #
    #     # Holds the intermediate graph representations
    #     global_representation = []
    #
    #     for i in range(self.n_layers):
    #         x = self.conv_layers[i](x, edge_index, edge_attr)
    #         x = torch.relu(self.transf_layers[i](x))
    #         x = self.bn_layers[i](x)
    #         # Always aggregate last layer
    #         if i % self.top_k_every_n == 0 or i == self.n_layers:
    #             x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
    #                 x, edge_index, edge_attr, batch_index
    #             )
    #             # Add current representation
    #             global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
    #
    #     x = sum(global_representation)
    #
    #     # Output block
    #     x = torch.relu(self.linear1(x))
    #     x = F.dropout(x, p=0.8, training=self.training)
    #     x = torch.relu(self.linear2(x))
    #     x = F.dropout(x, p=0.8, training=self.training)
    #     x = self.linear3(x)
    #
    #     return x

    def mse_loss(self, prediction, target):
        prediction = prediction.reshape(target.shape)
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
        return {'val_loss' : loss}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        prediction = self.forward(x)
        loss = self.mse_loss(prediction, y)
        self.log('test_loss', loss)
        return {'test_loss' : loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_results = {'test_loss' : avg_loss}
        return self.test_results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

class GNNDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        if 'data_dir' in config:
            self.data_dir = config['data_dir']
        else:
            # DOESN'T WORK WITH RAYTUNE, raytune executes in a different folder than where the python files are so explicitly pass the file location every time
            self.data_dir = 'C:\\Users\\bwvan\\PycharmProjects\\GenMol\\data\\'
        self.prepare_data_per_node = True
        self.val_batch_size = config['val_batch_size']
        self.train_batch_size = config['train_batch_size']
        self.collate_func = collate_for_graphs
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

def plot_train_and_val_using_mpl(train_loss, val_loss, name=None, save=False):
    """
    Plots the train and validation loss using Matplotlib
    """
    assert len(train_loss) == len(val_loss)

    f, ax = plt.subplots()
    x = np.arange(len(train_loss))
    ax.plot(x, np.array(train_loss), label='train')
    ax.plot(x, np.array(val_loss), label='val')
    ax.legend()
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    if save:
        if name != None:
            ax.set_title(name)
        else:
            ax.set_title('Anonymous plot')
        plt.savefig(name + '.png')
    return f, ax
