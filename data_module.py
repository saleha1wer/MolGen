import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from utils.from_smiles import from_smiles


print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
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
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule into PyG graph object Data
            data = from_smiles(mol['SMILES'])
            data.y = self._get_label(mol['pchembl_value_Mean'])
            if self.test:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_test_{index}.pt'))
            else:
                torch.save(data,
                           os.path.join(self.processed_dir,
                                        f'data_{index}.pt'))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

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
        self.val_batch_size = config['val_batch_size']
        self.train_batch_size = config['train_batch_size']
        self.num_workers = config['num_workers']
        self.train_data = data_train
        self.test_data = data_test

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataloader = DataLoader(dataset=self.train_data,
                                      batch_size=self.train_batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_dataloader = DataLoader(dataset=self.test_data,
                                    batch_size=self.val_batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers)
        return val_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader
