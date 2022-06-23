import os
import numpy as np
import pytorch_lightning as pl
import optuna
import time
from sklearn.model_selection import train_test_split
from functools import partial

from network import GNN_GINE
from data_module import GNNDataModule, MoleculeDataset
from utils.GINE_network import GINE

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.bohb import TuneBOHB
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray.tune.utils import wait_for_gpu

raytune_callback = TuneReportCheckpointCallback(
    metrics={
        'loss': 'val_loss'
    },
    filename='checkpoint',
    on='validation_end')


def main():
    adenosine_star = False
    max_epochs = 100
    n_samples = 100
    max_t_per_trial = 1000
    batch_size = 64

    # for prot_target_encoding choose: None or 'one-hot-encoding'
    prot_target_encoding = 'one-hot-encoding'

    if adenosine_star:
        dataset = MoleculeDataset(root=os.getcwd() + '/data/adenosine', filename='human_adenosine_ligands',
                                  prot_target_encoding=prot_target_encoding)
    else:
        dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',
                                  prot_target_encoding=prot_target_encoding)

    train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.8, random_state=0)

    data_train = dataset[train_indices.tolist()]
    data_test = dataset[test_indices.tolist()]

    datamodule_config = {
        'batch_size': batch_size,
        'num_workers': 8
    }

    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    grid_config = {
        'N': tune.grid_search([1, 3, 5, 9]),
        'E': tune.grid_search([0]),
        'lr': 3e-4,  # learning rate
        'hidden': 128,  # embedding/hidden dimensions
        'layer_type': GINE,
        'n_layers': 4,
        'num_workers': 8,
        'pool': 'mean',
        'v2': True,
        'input_heads': 1,
        'second_input': 'prot',
        'active_layer': 'first',
        'dropout_rate': 0.5,
        'batch_size': batch_size
        # 'batch_size': tune.choice([16,32,64,128])
    }


    # model = GNN(gnn_debug_config)
    # trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=200)

    def train_tune(config):
        model = GNN_GINE(config)
        trainer = pl.Trainer(max_epochs=max_epochs,
                             accelerator='cpu',
                             devices=1,
                             enable_progress_bar=True,
                             progress_bar_refresh_rate=0,
                             enable_checkpointing=True,
                             callbacks=[raytune_callback])
        trainer.fit(model, data_module)

    analysis = tune.run(partial(train_tune),
                        config=grid_config,
                        num_samples=n_samples,  # number of samples taken in the entire sample space
                        # scheduler=bohb_scheduler,
                        local_dir=os.getcwd(),
                        resources_per_trial={
                            'cpu': 8
                            # 'memory'    :   10 * 1024 * 1024 * 1024
                        })

    print('Finished grid search optimization.')

    best_configuration = analysis.get_best_config(metric='loss', mode='min', scope='last')
    best_trial = analysis.get_best_trial(metric='loss', mode='min', scope='last')

    print(f"Best trial configuration:{best_trial.config}")
    print(f"Best trial final validation loss:{best_trial.last_result['loss']}")



if __name__ == '__main__':
    main()
