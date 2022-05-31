import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset


# Dataloader class
# Source: https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
class GNNDataModule(pl.LightningDataModule):
    def __init__(self, config, df_train, df_test):
        super().__init__()
        self.prepare_data_per_node = True
        self.val_batch_size = config['val_batch_size']
        self.train_batch_size = config['train_batch_size']
        self.num_workers = config['num_workers']
        self.train_data = GraphDataSet(data=df_train['graphs'].to_numpy(), targets=df_train['pchembl_value_Mean'].to_numpy())
        self.test_data = GraphDataSet(data=df_test['graphs'].to_numpy(), targets=df_test['pchembl_value_Mean'].to_numpy())

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
