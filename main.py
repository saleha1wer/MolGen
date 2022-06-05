#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from functools import partial
# from rdkit import Chem
# from rdkit.Chem import PandasTools
# from utils.mol2fingerprint import calc_fps
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
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

    config = dict(node_feature_dimension=len(Graphs.ATOM_FEATURIZER.indx2atm),
        num_propagation_steps=tune.randint(1, 10),
        embeddings_dimension=len(Graphs.ATOM_FEATURIZER.indx2atm),
        train_batch_size=64,
        val_batch_size=64,
        num_workers=0,
#        data_dir='C:\\Users\\bwvan\\PycharmProjects\\GenMol\\data\\', removed since it broke the automatic naming function for the logs
        learning_rate=tune.choice([0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
        momentum=tune.uniform(0.001, 1.0),
        weight_decay=tune.loguniform(0.00001, 1.0),
        max_epochs=10)

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

    analysis = tune.run(partial(train_tune),
            config=config,
            num_samples=10,
            search_alg=search_alg,
#            progress_reporter=reporter,
#            scheduler=scheduler,
            local_dir='C:\\Users\\bwvan\\PycharmProjects\\GenMol\\data\\',
                        resources_per_trial={
                            'gpu'   :   1
                            #'memory'    :   10 * 1024 * 1024 * 1024
                        })


    print('Done with hyperparameter optimization.')
#    results_df = analysis.results_df
#    logdir = analysis.get_best_logdir('mean_accuracy')


if __name__ == '__main__':
    main()