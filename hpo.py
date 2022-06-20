
import os
import numpy as np
import pytorch_lightning as pl
import optuna
import time
from sklearn.model_selection import train_test_split
from functools import partial

from network import GNN
from data_module import GNNDataModule, MoleculeDataset

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

def run_hpo_finetuning(pretrain_epochs, finetune_epochs, n_samples, max_t_per_trial, pre_data_module, fine_data_module, gnn_config):
    def train_tune(config):
        pretrain_model = GNN(config)
        trainer = pl.Trainer(max_epochs=pretrain_epochs,
                             accelerator = config['accelerator'],
                             devices=1,
                             enable_progress_bar=True,
                             enable_checkpointing=True,
                             callbacks=[raytune_callback])
        trainer.fit(pretrain_model, pre_data_module)
        finetuned_model = finetune(source_model = pretrain_model, data_module = fine_data_module, epochs=finetune_epochs, patience=15)

    start_time = time.time()
    reporter = CLIReporter(parameter_columns=['learning_rate'],
                           metric_columns=['loss', 'training_iteration']
                           )

    bohb_scheduler = HyperBandForBOHB(
        time_attr='training_iteration',
        max_t=max_t_per_trial,
        metric='loss',
        mode='min'
    )

    bohb_search_alg = TuneBOHB(
        metric='loss',
        mode='min'
    )

    analysis = tune.run(partial(train_tune),
                        config=gnn_config,
                        num_samples=n_samples,  # number of samples taken in the entire sample space
                        search_alg=bohb_search_alg,
                        scheduler=bohb_scheduler,
                        local_dir=os.getcwd(),
                        resources_per_trial={
                            gnn_config['accelerator'] : 1
                            # 'memory'    :   10 * 1024 * 1024 * 1024
                        })

    print('Finished hyperparameter optimization.')
    best_configuration = analysis.get_best_config(metric='loss', mode='min', scope='last')
    best_trial = analysis.get_best_trial(metric='loss', mode='min', scope='last')

    print(f"Best trial configuration:{best_trial.config}")
    print(f"Best trial final validation loss:{best_trial.last_result['loss']}")

    #    print(f"attempting to load from dir: {best_trial.checkpoint.value}")
    #    print(f"attempting to load file: {best_trial.checkpoint.value + 'checkpoint'}")

    best_checkpoint_model = GNN.load_from_checkpoint(best_trial.checkpoint.value + '/checkpoint')

    best_config_model = GNN(best_configuration)

    #    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=gnn_config['accelerator'],
                            devices=1,
                            enable_progress_bar=True,
                            enable_checkpointing=True,
                            callbacks=[raytune_callback])

    test_data_loader = data_module.test_dataloader()

    test_results = trainer.test(best_config_model, test_data_loader)
    end_time = time.time()
    print(f"Elapsed time:{end_time - start_time}")
    return best_configuration, best_model

def run_hpo_basic(max_epochs, n_samples, max_t_per_trial, data_module, gnn_config):
    def train_tune(config):
        model = GNN(config)
        trainer = pl.Trainer(max_epochs=max_epochs,
                                accelerator=gnn_config['accelerator'],
                                devices=1,
                                enable_progress_bar=True,
                                enable_checkpointing=True,
                                callbacks=[raytune_callback])
        trainer.fit(model, data_module)

    start = time.time()
    # trainer.test(model, data_module) #loads the best checkpoint automatically
    reporter = CLIReporter(parameter_columns=['learning_rate', 'num_propagation_steps', 'weight_decay'],
                            metric_columns=['loss', 'training_iteration']
                            )

    bohb_scheduler = HyperBandForBOHB(
        time_attr='training_iteration',
        max_t=max_t_per_trial,
        metric='loss',
        mode='min'
    )

    bohb_search_alg = TuneBOHB(
        metric='loss',
        mode='min'
    )

    analysis = tune.run(partial(train_tune),
                        config=gnn_config,
                        num_samples=n_samples,  # number of samples taken in the entire sample space
                        search_alg=bohb_search_alg,
                        scheduler=bohb_scheduler,
                        local_dir=os.getcwd(),
                        resources_per_trial={
                            gnn_config['accelerator'] : 1
                            # 'memory'    :   10 * 1024 * 1024 * 1024
                        })

    print('Finished hyperparameter optimization.')
    best_configuration = analysis.get_best_config(metric='loss', mode='min', scope='last')
    best_trial = analysis.get_best_trial(metric='loss', mode='min', scope='last')

    print(f"Best trial configuration:{best_trial.config}")
    print(f"Best trial final validation loss:{best_trial.last_result['loss']}")

    #    print(f"attempting to load from dir: {best_trial.checkpoint.value}")
    #    print(f"attempting to load file: {best_trial.checkpoint.value + 'checkpoint'}")

    best_checkpoint_model = GNN.load_from_checkpoint(best_trial.checkpoint.value + '/checkpoint')

    best_config_model = GNN(best_configuration)

    #    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=gnn_config['accelerator'],
                            devices=1,
                            enable_progress_bar=True,
                            enable_checkpointing=True,
                            callbacks=[raytune_callback])

    test_data_loader = data_module.test_dataloader()

    test_results = trainer.test(best_config_model, test_data_loader)
    end = time.time()
    print(f"Elapsed time:{end - start}")
    return best_configuration