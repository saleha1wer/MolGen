from functools import partial
from utils.mol2graph import Graphs
import torch
from ray import tune
from network import GNN, GNNDataModule, collate_for_graphs
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

raytune_callback = TuneReportCallback(
    {
        'loss': 'val_loss',
        'mean_accuracy': 'val_accuracy'
    },
    on='validation_end')


def train_tune(config, max_epochs=10, gpus=0):
    datamodule = config.pop('datamodule')  # little hack..
    model = GNN(config)

    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator='gpu',
                         devices=1,
                         enable_progress_bar=True,
                         enable_checkpointing=True,
                         callbacks=[raytune_callback])
    trainer.fit(model, datamodule)

    # trainer.test(model, data_module) #loads the best checkpoint automatically


def main():
    torch.set_default_dtype(torch.float64)

    datamodule_config = {
        'collate_func': collate_for_graphs,
        'train_batch_size': 64,
        'val_batch_size': 64,
        'num_workers': 0,
        # 'data_dir': '/Users/Jurren/Documents/GitHub/GenMol/data/'
        'data_dir': '/home/molgen/GenMol/data/'  # full directory to data
    }
    datamodule = GNNDataModule(datamodule_config)

    config = {
        'learning_rate': tune.grid_search([1e-3, 3e-3, 1e-2]),
        'node_feature_dimension': len(Graphs.ATOM_FEATURIZER.indx2atm),
        'num_propagation_steps': tune.grid_search([3, 4]),
        'embeddings_dimension': len(Graphs.ATOM_FEATURIZER.indx2atm),
        'datamodule': datamodule
    }

    tune.run(partial(train_tune, max_epochs=100, gpus=1),
             config=config,
             local_dir=datamodule_config['data_dir'],
             num_samples=1)


if __name__ == '__main__':
    main()
