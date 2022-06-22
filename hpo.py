
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
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray.tune.utils import wait_for_gpu
from finetune import finetune
from datetime import datetime

test_pretraining_epochs = 150
test_finetuning_epochs = 100

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

def save_loss_and_config(val_loss, test_loss, configuration):

    now_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    file = open("HPO_result_logs/"+now_string+"-HPO_result.txt", 'w')
    message = f"Test loss achieved: {str(test_loss)} \nVal loss achieved:{str(val_loss)} \nConfiguration found: {str(configuration)}"
    file.write(message)

def meta_hpo_finetuning(pretrain_epochs, finetune_epochs, n_samples, train_size, report_test_loss = True):
    adenosine_star = False
    NUM_NODE_FEATURES = 5
    NUM_EDGE_FEATURES = 1
    max_t_per_trial = 2000  # hpo param
    batch_size = 64
    no_a2a = True  # use a2a data or not in adenosine set
    no_a2a = '_no_a2a' if no_a2a else ''
    # for prot_target_encoding choose: None or 'one-hot-encoding'
    # if choosing one-hot-encoding change input_heads in gnn_config

    #
    ######################################################################################################################
    # HPO on pretrain data (adenosine)
    # best_config = run_hpo(max_epochs=max_epochs, n_samples=n_samples, max_t_per_trial=max_t_per_trial, data_module=pre_data_module, gnn_config=gnn_config)
    # print('BEST CONFIG: ')
    # print(best_config)
    # Pretrain best config on pretrain data
    gnn_config = {
        'N': NUM_NODE_FEATURES,
        'E': NUM_EDGE_FEATURES,
        'lr': tune.loguniform(1e-4, 1e-2),  # learning rate
        'hidden': tune.choice([128, 256, 512]),  # embedding/hidden dimensions
        # 'layer_type': tune.choice([GIN, GAT, GraphSAGE]),
        'layer_type': GIN,
        'n_layers': tune.choice([2, 4, 6, 8]),
        'pool': tune.choice(['mean', 'GlobalAttention']),
        'accelerator': 'cpu',
        'batch_size': batch_size,
        'input_heads': 1,
        'active_layer': 'last',  # tune.choice(['first', 'last']),
        'trade_off_backbone': tune.loguniform(1e-4, 1e-2),
        'trade_off_head': tune.loguniform(1e-4, 1e-2),
        'order': tune.choice([1, 2]),  # is this a good interval?
        'patience': 10,
        'dropout_rate': tune.uniform(0, 0.6)
        # 'batch_size': tune.choice([16,32,64,128])
    }
    pre_datamodule, fine_datamodule = create_pretraining_finetuning_DataModules(batch_size, no_a2a, train_size)

    best_configuration, best_val_loss = run_hpo_finetuning(pretrain_epochs, finetune_epochs, n_samples, max_t_per_trial,
                                                       pre_datamodule, fine_datamodule, gnn_config)
    test_result = calculate_test_loss(pre_datamodule, fine_datamodule, pretrain_epochs, finetune_epochs, best_configuration)

    return best_val_loss, test_result[0]['test_loss'], best_configuration

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
    return test_result


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
        finetuned_model = finetune(save_model_name = 'final_',
                                   source_model = pretrain_model,
                                   data_module = fine_data_module,
                                   epochs=finetune_epochs,
                                   patience=config['patience'],
                                   trade_off_backbone=config['trade_off_backbone'],
                                   trade_off_head=config['trade_off_head'],
                                   order=config['order'],
                                   report_to_raytune=True)

    start_time = time.time()
    reporter = CLIReporter(parameter_columns=['learning_rate'],
                           metric_columns=['loss', 'training_iteration']
                           )

    optuna_search = OptunaSearch(
        metric='loss',
        mode='min'
    )

    bohb_scheduler = HyperBandForBOHB(
        time_attr='time_total_s',
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
                        search_alg=optuna_search,
                        #scheduler=bohb_scheduler,
                        local_dir=os.getcwd(),
                        trial_dirname_creator=trial_name_generator,
                        #memory=1 * 1024 * 1024 * 1024
#                                         resources_per_trial={
# #                            gnn_config['accelerator'] : 1,
#                             'memory'    :   1 * 1024 * 1024 * 1024
                        )

    print('Finished hyperparameter optimization.')
    best_configuration = analysis.get_best_config(metric='loss', mode='min', scope='last')
    best_trial = analysis.get_best_trial(metric='loss', mode='min', scope='last')

    print(f"Best trial configuration:{best_trial.config}")
    print(f"Best trial final validation loss:{best_trial.last_result['loss']}")

    end_time = time.time()
    print(f"Elapsed time:{end_time - start_time}")
    return best_configuration, best_trial.last_result['loss']

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