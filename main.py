import os
import numpy as np
import pytorch_lightning as pl
import optuna
import time
from copy import deepcopy
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
<<<<<<< HEAD
from hpo import run_hpo
from finetune import finetune

def main():
    pretrain_epochs = 100
    finetune_epochs = 75

    adenosine_star = False
    NUM_NODE_FEATURES = 9
    NUM_EDGE_FEATURES = 3
    max_epochs = 200 #hpo param
    n_samples = 100 #hpo param
    max_t_per_trial = 1000 #hpo param
    batch_size = 64 
    no_a2a = True #use a2a data or not in adenosine set
    no_a2a = '_no_a2a' if no_a2a else ''
    # for prot_target_encoding choose: None or 'one-hot-encoding'
    # if choosing one-hot-encoding change input_heads in gnn_config

    p_dataset = MoleculeDataset(root=os.getcwd() + '/data/adenosine{}'.format(no_a2a), filename='human_adenosine{}_ligands'.format(no_a2a),
                                prot_target_encoding=None)

    f_dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',

                                prot_target_encoding=None)
    all_train = []
    all_test = []
    for dataset in [p_dataset, f_dataset]:
        train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.8, random_state=0)
        data_train = dataset[train_indices.tolist()]
        data_test = dataset[test_indices.tolist()]
        all_train.append(data_train), all_test.append(data_test)
    p_data_train = all_train[0]
    p_data_test = all_test[0]
    f_data_train = all_train[1]
    f_data_test = all_test[1]
        
    datamodule_config = {
        'batch_size': batch_size,
        'num_workers': 0
    }
    pre_data_module = GNNDataModule(datamodule_config, p_data_train, p_data_test)
    fine_data_module = GNNDataModule(datamodule_config, f_data_train, f_data_test)

    gnn_config = {
        'N': NUM_NODE_FEATURES,
        'E': NUM_EDGE_FEATURES,
        'lr': tune.loguniform(1e-4, 1e-1),  # learning rate
        'hidden': tune.choice([16, 32, 64, 128, 256, 512, 1024]),  # embedding/hidden dimensions
        # 'layer_type': tune.choice([GIN, GAT, GraphSAGE]),
        'layer_type': GIN,
        'n_layers': tune.choice([2, 3, 4, 5, 6, 7]),
        'pool': tune.choice(['mean', 'GlobalAttention']),
        'batch_size': batch_size,
        'input_heads': 1
        # 'batch_size': tune.choice([16,32,64,128])
    }
    ######################################################################################################################
    # HPO on pretrain data (adenosine)
    best_config = run_hpo(max_epochs=max_epochs, n_samples=n_samples, max_t_per_trial=max_t_per_trial, data_module=pre_data_module, gnn_config=gnn_config)
    
    # Pretrain best config on pretrain data
    model = GNN(best_config)
    trainer = pl.Trainer(max_epochs=pretrain_epochs,
                                accelerator='cpu',
                                devices=1,
                                enable_progress_bar=True)
    trainer.fit(model, pre_data_module)

    # Finetune 
    source_model = model 
    finetune(save_model_name='final_', source_model=source_model, data_module=fine_data_module,epochs=finetune_epochs)

=======
from hyperparameter_optimization import hyperparameter_optimization

raytune_callback = TuneReportCheckpointCallback(
    metrics={
        'loss': 'val_loss'
    },
    filename='checkpoint',
    on='validation_end')


def main():
    results, elapsed_time = hyperparameter_optimization(max_epochs = 3,
                                                        n_samples = 2)
    print(f"printing results:{results}")
>>>>>>> 85811f3a53cab085ea4c665cb3199feccb00f723


if __name__ == '__main__':
    main()
