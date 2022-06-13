#import pandas as pd
#import numpy as np
import os
#import matplotlib.pyplot as plt
from functools import partial
# from rdkit import Chem
# from rdkit.Chem import PandasTools
# from utils.mol2fingerprint import calc_fps
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
from pandas import DataFrame
from utils.mol2graph import Graphs
import torch
from network import GNN, GNNDataModule, collate_for_graphs
import pytorch_lightning as pl

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.utils import wait_for_gpu
import optuna
import time

raytune_callback = TuneReportCheckpointCallback(
    metrics={
        'loss' : 'val_loss'
    },
    filename='checkpoint',
    on='validation_end')

def train_tune(config, checkpoint_dir=None):
    model = GNN(config)
    datamodule = GNNDataModule(config)

    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         accelerator='gpu',
                         devices=1,
                         enable_progress_bar=True,
                         enable_checkpointing=True,
                         callbacks=[raytune_callback])
    trainer.fit(model, datamodule)
#    trainer.test(model, data_module) #loads the best checkpoint automatically


def main():
    torch.set_default_dtype(torch.float64)
    node_feature_dimension = len(Graphs.ATOM_FEATURIZER.indx2atm)

    GNN_config = dict(
        train_batch_size=64,
        val_batch_size=64,
        num_workers=0)

    DataModule_config = dict(node_feature_dimension=len(Graphs.ATOM_FEATURIZER.indx2atm),
        num_propagation_steps=tune.randint(1, 15),
        embeddings_dimension=len(Graphs.ATOM_FEATURIZER.indx2atm),
        learning_rate=tune.loguniform(0.0001, 0.7),
        momentum=tune.loguniform(0.001, 1.0),
        weight_decay=tune.loguniform(0.00001, 1.0),
        max_epochs=30)

#    scheduler = ASHAScheduler(
#       max_t=50,
#       grace_period=1,
#       reduction_factor=2
#   )

    reporter = CLIReporter(
#        parameter_columns=['learning_rate', 'num_propagation_steps', 'weight_decay'],
        metric_columns=['loss', 'training_iteration']
    )

    search_alg = OptunaSearch(
        metric='loss',
        mode='min'
    )

    def join_dicts(a, b):
        return dict(list(a.items()) + list(b.items()))

    joined_config = join_dicts(GNN_config, DataModule_config)

    start = time.time()
    print(f"Starttime:{start}")
    analysis = tune.run(partial(train_tune),
            config=joined_config,
            num_samples=10, # number of samples taken in the entire sample space
            search_alg=search_alg,
#            progress_reporter=reporter,
#            scheduler=scheduler,
            local_dir='C:\\Users\\bwvan\\PycharmProjects\\GenMol\\tensorboardlogs\\',
                        resources_per_trial={
                            'gpu'   :   1
                            #'memory'    :   10 * 1024 * 1024 * 1024
                        })


    print('Finished with hyperparameter optimization.')
    best_configuration = analysis.get_best_config(metric='loss', mode='min', scope='last')
    best_trial = analysis.get_best_trial(metric='loss', mode='min', scope='last')

    print(f"Best trial configuration:{best_trial.config}")
    print(f"Best trial final validation loss:{best_trial.last_result['loss']}")

#    print(f"attempting to load from dir: {best_trial.checkpoint.value}")
#    print(f"attempting to load file: {best_trial.checkpoint.value + 'checkpoint'}")

    test_config = join_dicts(best_configuration, DataModule_config)

    best_checkpoint_model = GNN.load_from_checkpoint(best_trial.checkpoint.value+'/checkpoint')

    test_datamodule = GNNDataModule(test_config)

    trainer = pl.Trainer(max_epochs=test_config['max_epochs'],
                         accelerator='gpu',
                         devices=1,
                         enable_progress_bar=True,
                         enable_checkpointing=True,
                         callbacks=[raytune_callback])
    test_results = trainer.test(best_checkpoint_model, test_datamodule)

    end = time.time()
    print(f"Elapsed time:{end-start}")
if __name__ == '__main__':
    main()