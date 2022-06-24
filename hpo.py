
import os
import numpy as np
import pytorch_lightning as pl
import optuna
import time
from sklearn.model_selection import train_test_split
from functools import partial

from network import GNN
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray.tune.utils import wait_for_gpu
from finetune import finetune
from datetime import datetime
import torch 

test_pretraining_epochs = 125
test_finetuning_epochs = 75

raytune_callback = TuneReportCheckpointCallback(
    metrics={
        'loss': 'val_loss'
    },
    filename='checkpoint',
    on='validation_end')

#this function makes a custom logging directory name since the normal way (concat all ...
# the parameters) makes the name too long for windows and it errors on creation of the logging dir
def trial_name_generator(trial):
    namestring = str(trial.config['N'])+str(trial.config['E'])+str(trial.config['hidden'])+str(trial.config['n_layers'])+str(trial.trial_id)
    return namestring

def save_loss_and_config(val_loss='', test_loss='', configuration=''):

    now_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    file = open("HPO_result_logs/"+now_string+"-HPO_result.txt", 'w')
    message = f"Test loss achieved: {str(test_loss)} \nVal loss achieved:{str(val_loss)} \nConfiguration found: {str(configuration)}"
    file.write(message)

def meta_hpo_basic(train_epochs, n_samples, train_size,space, report_test_loss = True):
    batch_size = 64
    no_a2a = True  # use a2a data or not in adenosine set
    no_a2a = '_no_a2a' if no_a2a else ''
    # for prot_target_encoding choose: None or 'one-hot-encoding'
    # if choosing one-hot-encoding change input_heads in gnn_config

    gnn_config = space
    # pre_datamodule, fine_datamodule = create_pretraining_finetuning_DataModules(batch_size, no_a2a, train_size)
    dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',
                                prot_target_encoding=None,xgb = None,include_fps=False)

    train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=train_size, random_state=0)
    data_train = dataset[train_indices.tolist()]
    data_test = dataset[test_indices.tolist()]

    datamodule_config = {
        'batch_size': batch_size,
        'num_workers': 0
    }
    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    best_configuration, best_val_loss, best_test_loss = run_hpo_basic(train_epochs, n_samples,
                                                       data_module, gnn_config)

    return best_val_loss,best_test_loss, best_configuration



def meta_hpo_finetuning(finetune_epochs, patience,n_samples, train_size,source_model, space,report_test_loss = True):
    batch_size = 64
    no_a2a = True  # use a2a data or not in adenosine set
    no_a2a = '_no_a2a' if no_a2a else ''
    # for prot_target_encoding choose: None or 'one-hot-encoding'
    # if choosing one-hot-encoding change input_heads in gnn_config

    gnn_config = space
    pre_datamodule, fine_datamodule = create_pretraining_finetuning_DataModules(batch_size, no_a2a, train_size)

    best_configuration, best_val_loss = run_hpo_finetuning(finetune_epochs, patience, n_samples,
                                                          fine_datamodule, gnn_config,source_model)

    return best_val_loss, best_configuration

def calculate_test_loss(pre_datamodule, finetune_data_module, config):
    pretrain_model = GNN(config)
    trainer = pl.Trainer(max_epochs=test_pretraining_epochs,
                         accelerator=config['accelerator'],
                         devices=1,
                         enable_progress_bar=True,
                         enable_checkpointing=True,
                         callbacks=[raytune_callback])
    trainer.fit(pretrain_model, pre_datamodule)
    finetuned_model = finetune(save_model_name='final_',
                               source_model=pretrain_model,
                               data_module=finetune_data_module,
                               epochs=test_finetuning_epochs,
                               patience=config['patience'],
                               trade_off_backbone=config['trade_off_backbone'],
                               trade_off_head=config['trade_off_head'],
                               order=config['order'],
                               report_to_raytune = False)
    test_result = trainer.test(finetuned_model, finetune_data_module)
    torch.save(finetuned_model.state_dict(), 'models/final_model')
    return test_result


def run_hpo_finetuning(finetune_epochs,patience, n_samples, fine_data_module, gnn_config, source_model):
    def train_tune(config):
        finetuned_model = finetune(save_model_name = 'final_',
                                   source_model = source_model,
                                   data_module = fine_data_module,
                                   epochs=finetune_epochs,
                                   patience=patience,
                                   trade_off_backbone=config['trade_off_backbone'],
                                   trade_off_head=config['trade_off_head'],
                                   order=config['order'],
                                   report_to_raytune=True)

    start_time = time.time()

    tpe = HyperOptSearch(
            metric="loss", mode="min", n_initial_points=10)

    analysis = tune.run(partial(train_tune),
                        config=gnn_config,
                        num_samples=n_samples,  # number of samples taken in the entire sample space
                        search_alg=tpe,
                        local_dir=os.getcwd(),
                        trial_dirname_creator=trial_name_generator)

    print('Finished hyperparameter optimization.')
    best_configuration = analysis.get_best_config(metric='loss', mode='min', scope='last')
    best_trial = analysis.get_best_trial(metric='loss', mode='min', scope='last')

    end_time = time.time()
    print(f"Elapsed time:{end_time - start_time}")
    return best_configuration, best_trial.last_result['loss']

def run_hpo_basic(max_epochs, n_samples, data_module, gnn_config):
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

    tpe = HyperOptSearch(
            metric="loss", mode="min", n_initial_points=10)

    analysis = tune.run(partial(train_tune),
                        config=gnn_config,
                        num_samples=n_samples,  # number of samples taken in the entire sample space
                        search_alg=tpe,
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

    #    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=gnn_config['accelerator'],
                            devices=1,
                            enable_progress_bar=True,
                            enable_checkpointing=True,
                            callbacks=[raytune_callback])

    val_data_loader = data_module.val_dataloader()
    test_data_loader = data_module.test_dataloader()
    val_results = trainer.test(best_checkpoint_model, val_data_loader)
    test_results = trainer.test(best_checkpoint_model, test_data_loader)
    end = time.time()
    print(f"Elapsed time:{end - start}")
    return best_configuration,val_results,test_results