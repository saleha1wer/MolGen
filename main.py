import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from functools import partial
# from xgboost import XGBRegressor

from network import GNN
from data_module import GNNDataModule, MoleculeDataset

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
        'loss': 'val_loss'
    },
    filename='checkpoint',
    on='validation_end')


def main():

    adenosine_star = False
    NUM_NODE_FEATURES = 1
    NUM_EDGE_FEATURES = 1
    # for the moment, loading the zip file does not
    # pd.DataFrame with 'SMILES' and 'pchembl_value_Mean'
    if adenosine_star:
        dataset = MoleculeDataset(root='data/adenosine', filename='human_adenosine_ligands')
    else:
        dataset = MoleculeDataset(root='data/a2aar', filename='human_a2aar_ligands')

    train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.8, random_state=0)

    data_train = dataset[train_indices]
    data_test = dataset[test_indices]

    batch_size = 64
    datamodule_config = {
        'train_batch_size': batch_size,
        'val_batch_size': batch_size,
        'num_workers': 0
    }

    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    gnn_config = {
        'learning_rate': 3e-3,
            # tune.grid_search([1e-3, 3e-3, 1e-2]),
        'node_feature_dimension': NUM_NODE_FEATURES,
        'edge_feature_dimension': NUM_EDGE_FEATURES,
        'num_propagation_steps': 4,
            #tune.grid_search([3, 4]),
        'embedding_dimension': 50,
            #tune.grid_search([64, 128])
    }

    model = GNN(gnn_config)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=200)

    trainer.fit(model, data_module)

    # def train_tune(config, checkpoint_dir=None):
    #     model = GNN(config)
    #
    #     trainer = pl.Trainer(max_epochs=config['max_epochs'],
    #                          accelerator='gpu',
    #                          devices=1,
    #                          enable_progress_bar=True,
    #                          enable_checkpointing=True,
    #                          callbacks=[raytune_callback])
    #     trainer.fit(model, datamodule)
    #
    # #    trainer.test(model, data_module) #loads the best checkpoint automatically
    # reporter = CLIReporter(
    #     #        parameter_columns=['learning_rate', 'num_propagation_steps', 'weight_decay'],
    #     metric_columns=['loss', 'training_iteration']
    # )
    #
    # search_alg = OptunaSearch(
    #     metric='loss',
    #     mode='min'
    # )

    # analysis = tune.run(partial(train_tune),
    #                     config=joined_config,
    #                     num_samples=6,  # number of samples taken in the entire sample space
    #                     search_alg=search_alg,
    #                     #            progress_reporter=reporter,
    #                     #            scheduler=scheduler,
    #                     local_dir='C:\\Users\\bwvan\\PycharmProjects\\GenMol\\tensorboardlogs\\',
    #                     resources_per_trial={
    #                         'gpu': 0
    #                         # 'memory'    :   10 * 1024 * 1024 * 1024
    #                     })
    #
    # print('Finished with hyperparameter optimization.')
    # best_configuration = analysis.get_best_config(metric='loss', mode='min', scope='last')
    # best_trial = analysis.get_best_trial(metric='loss', mode='min', scope='last')
    #
    # print(f"Best trial configuration:{best_trial.config}")
    # print(f"Best trial final validation loss:{best_trial.last_result['loss']}")
    #
    # #    print(f"attempting to load from dir: {best_trial.checkpoint.value}")
    # #    print(f"attempting to load file: {best_trial.checkpoint.value + 'checkpoint'}")
    #
    # test_config = join_dicts(best_configuration, DataModule_config)
    #
    # best_checkpoint_model = GNN.load_from_checkpoint(best_trial.checkpoint.value + '/checkpoint')
    #
    # test_datamodule = GNNDataModule(test_config)
    #
    # trainer = pl.Trainer(max_epochs=test_config['max_epochs'],
    #                      accelerator='gpu',
    #                      devices=1,
    #                      enable_progress_bar=True,
    #                      enable_checkpointing=True,
    #                      callbacks=[raytune_callback])
    # test_results = trainer.test(best_checkpoint_model, test_datamodule)
    #
    # end = time.time()
    # print(f"Elapsed time:{end - start}")


if __name__ == '__main__':
    main()
