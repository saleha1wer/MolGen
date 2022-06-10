import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor

from network import GNN
from data_module import GNNDataModule, MoleculeDataset


def main():

    adenosine_star = False
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
        'num_workers': 1
    }

    data_module = GNNDataModule(datamodule_config, data_train, data_test)

    gnn_config = {
        'learning_rate': 3e-3,
        'node_feature_dim': 1,
        'edge_dim': 1,
        'embedding_dim': 64
    }

    model = GNN(gnn_config)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=50)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
