import pandas as pd
import os
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from utils.from_smiles import from_smiles
from utils.encode_protein import prot_target2array, prot_target2one_hot

from sklearn.model_selection import train_test_split

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


# https://github.com/deepfindr/gnn-project/blob/main/dataset_featurizer.py
class MoleculeDataset(Dataset):
    def __init__(self, root, filename, prot_target_encoding=None, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        self.prot_target_encoding = prot_target_encoding
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.prot_target_encoding != None:
            prot_target_encoder = self._target_encoder(self.prot_target_encoding)
        # self.data = self.data.head(400)  # for debugging purposes
        self.length = self.data.shape[0]
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule into PyG graph object Data
            data = from_smiles(mol['SMILES'])
            # wrapping the value in a list to have the same shape as the network prediction
            data.y = torch.tensor([[mol['pchembl_value_Mean']]])
            if self.prot_target_encoding != None:
                data.p = prot_target_encoder(mol['target_id'])
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))

    def _target_encoder(self, target_encoding):
        if target_encoding == 'one-hot-encoding':
            return prot_target2one_hot

    # When implementing two-to-one:
    #   Decide how to save encoded protein targets,
    #   this could be added to the above loop
    #         protein_targets = []
    #         encoded = dict()
    #         count = 0
    #         for target in mol['target_id']:
    #             if target in encoded.keys():
    #                 array = encoded[target]
    #             else:
    #                 array = prot_target2array(target)
    #                 encoded[target] = array
    #             protein_targets.append(array)
    #         print('Finished encoding protein targets')
    #         protein_targets = np.array(protein_targets)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'data_{idx}.pt'))
        return data


# Dataloader class
# Source: https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
class GNNDataModule(pl.LightningDataModule):
    def __init__(self, config, data_train, data_test):
        super().__init__()
        self.prepare_data_per_node = True
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        # all_data = data_train + data_test
        data_train, data_val = train_test_split(data_train, test_size=0.1, random_state=0)  # TODO add a randomstate
        self.train_data = data_train
        self.val_data = data_val
        self.test_data = data_test

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataloader = DataLoader(dataset=self.train_data,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataloader = DataLoader(dataset=self.val_data,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers)
        return val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataloader = DataLoader(dataset=self.test_data,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers)
        return test_dataloader

    def all_dataloader(self) -> EVAL_DATALOADERS:
        train_dev_sets = torch.utils.data.ConcatDataset([self.train_data, self.test_data,self.val_data])
        all_dataloader = DataLoader(dataset=train_dev_sets,   #make this legnth the length of the dataset
                                     batch_size=train_dev_sets.__len__(),
                                     shuffle=False,
                                     num_workers=self.num_workers)
        return all_dataloader

