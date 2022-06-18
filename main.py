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


def main():
    adenosine_star = False
    NUM_NODE_FEATURES = 9
    NUM_EDGE_FEATURES = 3
    max_epochs = 200
    n_samples = 100
    max_t_per_trial = 1000
    batch_size = 64

    # for prot_target_encoding choose: None or 'one-hot-encoding'
    # if choosing one-hot-encoding change input_heads in gnn_config
    if adenosine_star:
        dataset = MoleculeDataset(root=os.getcwd() + '/data/adenosine', filename='human_adenosine_ligands',
                                  prot_target_encoding=None)
    else:
        dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',
                                  prot_target_encoding=None)


    train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.8, random_state=0)

    data_train = dataset[train_indices.tolist()]
    data_test = dataset[test_indices.tolist()]

    datamodule_config = {
        'batch_size': batch_size,
        'num_workers': 0
    }

    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    gnn_config = {
        'N': NUM_NODE_FEATURES,
        'E': NUM_EDGE_FEATURES,
        'lr': tune.loguniform(1e-4, 1e-1),  # learning rate
        'hidden': tune.choice([16, 32, 64, 128, 256, 512, 1024]),  # embedding/hidden dimensions
        'layer_type': tune.choice([GIN, GAT, GraphSAGE]),
        'n_layers': tune.choice([2, 3, 4, 5, 6, 7]),
        'pool': tune.choice(['mean', 'GlobalAttention']),
        'batch_size': batch_size,
        'input_heads': 1
        # 'batch_size': tune.choice([16,32,64,128])
    }

    def train_tune(config, checkpoint_dir=None):
        model = GNN(config)
        trainer = pl.Trainer(max_epochs=max_epochs,
                             accelerator='cpu',
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

    optuna_search_alg = OptunaSearch(
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
                            'cpu': 1
                            # 'memory'    :   10 * 1024 * 1024 * 1024
                        })

    print('Finished with hyperparameter optimization.')
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
                         accelerator='cpu',
                         devices=1,
                         enable_progress_bar=True,
                         enable_checkpointing=True,
                         callbacks=[raytune_callback])

    test_data_loader = data_module.test_dataloader()

    test_results = trainer.test(best_config_model, test_data_loader)
    end = time.time()
    print(f"Elapsed time:{end - start}")


if __name__ == '__main__':
    main()
