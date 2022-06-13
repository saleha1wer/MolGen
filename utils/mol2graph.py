# TAKEN FROM JOHN BRADSHAW TUTORIAL - ML FOR MOLECULES ----- https://github.com/john-bradshaw/ml-in-bioinformatics-summer-school-2020
# TRANSLATES RDKIT.MOL OBJECTS TO GRAPHS

import torch

import numpy as np
import pandas as pd
from rdkit import Chem
import typing

class SymbolFeaturizer:
    """
    Symbol featurizer takes in a symbol and returns an array representing its
    one-hot encoding.
    """

    def __init__(self, symbols, feature_size=None):
        self.atm2indx = {k: i for i, k in enumerate(symbols)}
        self.indx2atm = {v: k for k, v in self.atm2indx.items()}
        self.feature_size = feature_size if feature_size is not None else len(symbols)

    def __call__(self, atom_symbol):
        out = np.zeros(self.feature_size)
        out[self.atm2indx[atom_symbol]] = 1.
        return out


class Graphs:
    ATOM_FEATURIZER = SymbolFeaturizer(['Ag', 'Al', 'Ar', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C',
                                        'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Eu', 'F',
                                        'Fe', 'Ga', 'Ge', 'H', 'He', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La',
                                        'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb',
                                        'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se',
                                        'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Te', 'Ti', 'Tl', 'V', 'W', 'Xe', 'Y',
                                        'Yb', 'Zn', 'Zr'])
    # ^ you can change the number of symbols here to play with the dimensionality,
    # we only need to have the symbols: ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']

    BOND_FEATURIZER = SymbolFeaturizer([1., 1.5, 2., 3.])

    # ^ single, aromatic, double and triple bonds (see earlier how RDKit represents these as doubles.)

    def __init__(self, node_features: torch.Tensor,
                 edge_list: torch.Tensor, edge_features: torch.Tensor, node_to_graph_id: torch.Tensor):
        """
        A graph datastructure which groups together the series of tensors that represent the
        graph. Note that this datastructure also holds multiple molecule graphs as one large
        disconnected graph -- the nodes belonging to each molecule are described by node_to_graph_id.

        ## Further details on the individual tensors
        Say this graph represents acetone, CC(=O)C, and ethane, CC, and we're using a simple
        three dimensional one-hot encoding for 'C', 'O' and 'N' and a simple two dimensional
        one-hot encoding for the bonds 'SINGLE', 'DOUBLE' then the resulting tensors would look
        like:
        node_features = [[1. 0. 0.],
                         [1. 0. 0.],
                         [0. 1. 0.],
                         [1. 0. 0.],
                         [1. 0. 0.],
                         [1. 0. 0.]]
        edge_list = [[0 1],
                     [1 0],
                     [1 2],
                     [2 1],
                     [1 3],
                     [3 1],
                     [4 5],
                     [5 4]]

        edge_features = [[1. 0.],
                         [1. 0.],
                         [0. 1.],
                         [0. 1.],
                         [1. 0.],
                         [1. 0.],
                         [1. 0.],
                         [1. 0.]]

        node_to_graph_id = [0 0 0 0 1 1]


        More generally we expect the different tensors to have the following datatypes and shapes
        (below N is number of nodes, E number of edges, h_n the feature dimensionality of node
        features and h_e the feature dimensionality of edge features):

        :param node_features: Tensor (dtype float32 , shape [N, h_n])
        :param edge_list: Tensor (dtype int64 , shape [E, 2])
        :param edge_features: Tensor (dtype float32 , shape [E, h_e])
        :param node_to_graph_id: Tensor (dtype int64 , shape [N]) this contains for each node
           the associated graph it belongs to. So for instance if this Graph datastructure
           represented only one graph this should be all zeros, however if two then it should be
           zeros for the nodes corresponding to the first graph and then 1s for the second graph.
           Graph ids should start at one and consist of consectutive integers.
        """
        self.node_features = node_features
        self.edge_list = edge_list
        self.edge_features = edge_features
        self.node_to_graph_id = node_to_graph_id

    def to(self, *args, **kwargs):
        """
        Works in a similar way to the Tensor function torch.Tensor.to(...)
        and performs  dtype and/or device conversion for the entire datastructure
        """
        new_graph = type(self)(self.node_features.to(*args, **kwargs),
                               self.edge_list.to(*args, **kwargs),
                               self.edge_features.to(*args, **kwargs),
                               self.node_to_graph_id.to(*args, **kwargs)
                               )
        return new_graph

    @classmethod
    def from_mol(cls, mol):
        """
        Converts a SMILES string into the representation required by this datastructure.
        """
        # Convert to form we need using previous code:
        node_features, edge_list, edge_features = mol_to_edge_list_graph(mol, cls.ATOM_FEATURIZER)
        edge_features = [cls.BOND_FEATURIZER(elem) for elem in edge_features]
        # ^ nb here we're converting the edge feature list into one-hot form

        # Convert to tensors:
        node_features = torch.tensor(node_features, dtype=torch.float64)
        edge_list = torch.tensor(edge_list, dtype=torch.int64)
        edge_features = torch.tensor(np.array(edge_features), dtype=torch.float64)
        node_to_graph_id = torch.zeros(node_features.shape[0], dtype=torch.int64)
        # ^we only (currently) have one molecule per SMILES so all the nodes can be assigned
        # the same id

        return cls(node_features, edge_list, edge_features, node_to_graph_id)

    @property
    def num_graphs(self):
        return torch.unique(self.node_to_graph_id).shape[0]

    @classmethod
    def concatenate(cls, list_of_graphs):
        """
        This takes in a list of objects of this class and joins them to form one large disconnected graph.

        For instance say we have two individual `Graphs` instances, one for acetone (CC(=O)C) and one for
        ethane (CC) they might look like this (in pseudocode -- note also in practice are one-hot encoding
        is larger):
        acetone = Graphs(
            node_features = [[1. 0. 0.],
                             [1. 0. 0.],
                             [0. 1. 0.],
                             [1. 0. 0.]]
            edge_list = [[0 1],
                         [1 0],
                         [1 2],
                         [2 1],
                         [1 3],
                         [3 1]]
            edge_features = [[1. 0.],
                             [1. 0.],
                             [0. 1.],
                             [0. 1.],
                             [1. 0.],
                             [1. 0.]]
            node_to_graph_id = [0 0 0 0]
        )

        ethane = Graphs(
            node_features = [[1. 0. 0.],
                             [1. 0. 0.]]
            edge_list = [[0 1],
                         [1 0]]
            edge_features = [[1. 0.],
                             [1. 0.]]
            node_to_graph_id = [0 0]
        )

        and this function would transform them into one large disconnected graph
        minibatch_of_graphs = Graphs(
                node_features = [[1. 0. 0.],
                                 [1. 0. 0.],
                                 [0. 1. 0.],
                                 [1. 0. 0.],
                                 [1. 0. 0.],
                                 [1. 0. 0.]]
                edge_list = [[0 1],
                             [1 0],
                             [1 2],
                             [2 1],
                             [1 3],
                             [3 1],
                             [4 5],
                             [5 4]]
                edge_features = [[1. 0.],
                                 [1. 0.],
                                 [0. 1.],
                                 [0. 1.],
                                 [1. 0.],
                                 [1. 0.],
                                 [1. 0.],
                                 [1. 0.]]
                node_to_graph_id = [0 0 0 0 1 1]
                )
        """
        # node features and edge_features simply get concatenated
        new_node_features = torch.cat([e.node_features for e in list_of_graphs], dim=0)
        new_edge_features = torch.cat([e.edge_features for e in list_of_graphs], dim=0)

        # edge_list and node
        new_edge_lists = []
        new_node_ids = []
        num_nodes_seen_so_far = 0
        num_graphs_so_far = 0
        for graph in list_of_graphs:
            new_edge_lists.append(graph.edge_list + num_nodes_seen_so_far)
            # ^ shift up the edges to reflect the nodes new indices

            new_node_ids.append(graph.node_to_graph_id + num_graphs_so_far)
            # ^shift up the node to graph id to reflect the number of graphs before

            num_nodes_seen_so_far += graph.node_features.shape[0]
            num_graphs_so_far += torch.unique(graph.node_to_graph_id).shape[0]

        new_edge_lists = torch.cat(new_edge_lists, dim=0)
        new_new_node_ids = torch.cat(new_node_ids, dim=0)

        new_concatenated_graph = cls(node_features=new_node_features,
                                     edge_list=new_edge_lists,
                                     edge_features=new_edge_features,
                                     node_to_graph_id=new_new_node_ids)
        return new_concatenated_graph



class GraphRegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that holds SMILES molecule data along with an associated single
    regression target.
    """

    def __init__(self, graph_list: Graphs,
                 regression_target_list: typing.List[float],
                 transform: typing.Optional[typing.Callable] = None):
        """
        :param graph_list: list of SMILES strings represnting the molecules
        we are regressing on.
        :param regression_target_list: list of targets
        :param transform: an optional transform which will be applied to the
        SMILES string before it is returned.
        """
        self.graph_list = graph_list
        self.regression_target_list = regression_target_list
        self.transform = transform

        assert len(self.graph_list) == len(self.regression_target_list), \
            "Dataset and targets should be the same length!"

    def __getitem__(self, index):
        x, y = self.graph_list[index], self.regression_target_list[index]
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.graph_list)

    @classmethod
    def create_from_df(cls, df: pd.DataFrame, graph_column: str = 'x',
                       regression_column: str = 'y', transform=None):
        """
        convenience method that takes in a Pandas dataframe and turns it
        into an   instance of this class.
        :param df: Dataframe containing the data.
        :param graph_column: name of column that contains the x data
        :param regression_column: name of the column which contains the
        y data (i.e. targets)
        :param transform: a transform to pass to class's constructor
        """
        graph_list = [x for x in df[graph_column].tolist()]
        targets = [float(y) for y in df[regression_column].tolist()]
        return cls(graph_list, targets, transform)


def mol_to_edge_list_graph(mol: Chem.Mol, atm_featurizer: SymbolFeaturizer) -> typing.Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Function that takes in a RDKit molecule (of N atoms, E bonds) and returns three numpy arrays:
    * the node features array (of dtype np.float32, shape [N, d]), which is a one hot encoding
    of the element of each atom type.
    * the edge list (of dtype np.int32, shape [2E, 2]) that represents the start and end index
    of each edge.
    * the edge feature list (of dtype np.float32, shape [2E, 1]) which describes the feature type
    associated with each edge.
    """
    # Node features
    node_features = [atm_featurizer(atm.GetSymbol()) for atm in mol.GetAtoms()]
    node_features = np.array(node_features, dtype=np.float32)

    # Edge list and edge feature list
    edge_list = []
    edge_feature_list = []
    for bnd in mol.GetBonds():
        bnd_indices = [bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()]
        bnd_type = bnd.GetBondTypeAsDouble()
        edge_list.extend([bnd_indices, bnd_indices[::-1]])
        edge_feature_list.extend([bnd_type, bnd_type])
    edge_list = np.array(edge_list, dtype=np.int32)
    edge_feature_list = np.array(edge_feature_list, dtype=np.float32)

    return node_features, edge_list, edge_feature_list


