import pandas as pd
import pytorch_lightning as pl
from rdkit import Chem
from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from network import GNN
from data_module import GNNDataModule
from utils.from_smiles import from_smiles


def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)

def processing(df : pd.DataFrame):
    df = df.dropna(axis=0)  # drop rows with missing values
    # smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
    #     .replace('[N]', 'N').replace('[B]', 'B') \
    #     .replace('[2H]', '[H]').replace('[3H]', '[H]')

    # doesn't assuming that all dupicates are removed already - Papyrus
    # df['Smiles'] = df['Smiles'].apply(canonical_smiles)
    # df = df.drop_duplicates(subset=['Smiles'])
    print('Finished preprocessing Smiles')

    print(df.shape)
    df['graphs'] = df['SMILES'].apply(from_smiles)

    # data selection and split
    df_train, df_test = train_test_split(df, test_size=0.2)
    print(df_train.shape)

    return df_train, df_test

    # PandasTools.AddMoleculeColumnToFrame(df, 'Smiles', 'Molecule', includeFingerprints=False)
    # print('Processed Smiles to Mol object')

    # FEATURIZATION OF MOLECULES
    # fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEx method)
    # print('Finished calculating fingerprints')

    # MAKING TRAIN AND TEST SET (validation?)
    # y_train, y_test, fps_train, fps_test, graphs_train, graphs_test = train_test_split(target_values, fps, graphs,
    #                                                                                    test_size=0.2)


def main():

    # for the moment, loading the zip file does not
    df = pd.read_csv('data/papyrus_ligand')

    # df = df.head(80)

    print(df.shape)
    print(df.columns)

    df = df[['SMILES', 'pchembl_value_Mean']]
    df_train, df_test = processing(df)

    batch_size = 64
    datamodule_config = {
        'train_batch_size': batch_size,
        'val_batch_size': batch_size,
        'num_workers': 1
    }

    data_module = GNNDataModule(datamodule_config, df_train, df_test)

    gnn_config = {
        'node_feature_dim': 9,
        'edge_dim' : 3,
        'embedding_dim': [64, 128]
    }

    model = GNN(gnn_config)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=10)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()