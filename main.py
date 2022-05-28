import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit.Chem import PandasTools
from utils.mol2fingerprint import calc_fps
from utils.preprocessing_to_smiles import canonical_smiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from utils.mol2graph import Graphs
from network import GNN, GNNDataModule, train_neural_network, plot_train_and_val_using_altair, collate_for_graphs, plot_train_and_val_using_mpl, TrainParams, DebuggingParams
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
def main():

    # # debugging code begin
    # df = pd.read_csv('DrugEx/data/LIGAND_RAW.tsv', sep='\t', header=0)
    # df = df[['Smiles', 'pChEMBL_Value']]
    # df = df.dropna(axis=0)  # drop rows with missing values
    # # smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
    # #     .replace('[N]', 'N').replace('[B]', 'B') \
    # #     .replace('[2H]', '[H]').replace('[3H]', '[H]')
    # df['Smiles'] = df['Smiles'].map(canonical_smiles)
    # df = df.drop_duplicates(subset=['Smiles'])
    # print('Finished preprocessing Smiles')
    #
    # PandasTools.AddMoleculeColumnToFrame(df, 'Smiles', 'Molecule', includeFingerprints=False)
    # print('Processed Smiles to Mol object')
    #
    # # REGRESSION TARGET VALUES
    # target_values = df['pChEMBL_Value'].to_numpy()
    #
    # # FEATURIZATION OF MOLECULES
    # fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEx method)
    # print('Finished calculating fingerprints')
    #
    # graphs = np.array([[]])  # TUTORIAL METHOD (JOHN BRADSHAW)
    # for mol in df['Molecule']:
    #     graphs = np.append(graphs, Graphs.from_mol(mol))
    # print('Finished making graphs')
    #
    #
    # y_train, y_test, fps_train, fps_test, graphs_train, graphs_test = train_test_split(target_values, fps, graphs,
    #                                                                                    test_size=0.2)
    # train_data = pd.DataFrame({'x': graphs_train, 'y': y_train})
    # test_data = pd.DataFrame({'x': graphs_test, 'y': y_test})
    #
    # train_data_alt = pd.concat([pd.DataFrame(graphs), pd.DataFrame(y_train)], axis=1)
    # train_data_alt_DL = DataLoader(train_data_alt, batch_size=19)
    #
    # print('graphs_train.shape:{}'.format(graphs_train.shape))
    # print('y_train.shape:{}'.format(y_train.shape))
    #
    # graphs_train = graphs_train.reshape(graphs_train.shape[0],1)
    # y_train = y_train.reshape(y_train.shape[0],1)
    #
    # graphs_test = graphs_test.reshape(graphs_test.shape[0],1)
    # y_test = y_test.reshape(y_test.shape[0],1)
    #
    # print('graphs_train.shape:{}'.format(graphs_train.shape))
    # print('y_train.shape:{}'.format(y_train.shape))
    #
    # train_data_complete = np.concatenate((graphs_train, y_train), axis=1)
    # test_data_complete = np.concatenate((graphs_test, y_test), axis=1)
    #
    # train_data_complete_DL = DataLoader(train_data_complete, batch_size=3)
    #
    # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    # mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    # mnist_train_DL = DataLoader(mnist_train, batch_size=7)
    # print(mnist_train)
    # print('mnist_train_DL ====================================')
    # print(type(mnist_train_DL))
    # print('mnist_train :{}'.format(mnist_train_DL))
    # for idx, item in enumerate(mnist_train_DL):
    #     print("idx:{}\t item:{}".format(idx, item))
    #     break
    #
    # print('train_data_complete ====================================')
    # print(type(train_data_complete))
    # print('train_data_complete.shape:{}'.format(train_data_complete.shape))
    # for idx, item in enumerate(train_data_complete):
    #     print("idx:{}\t item:{}".format(idx, item))
    #     break
    #
    # print('train_data_complete_DL ====================================')
    # print(type(train_data_complete_DL))
    # for idx, item in enumerate(train_data_complete_DL):
    #     print("idx:{}\t item:{}".format(idx, item))
    #     break
    # debugging code end

    torch.set_default_dtype(torch.float64)
    node_feature_dimension = len(Graphs.ATOM_FEATURIZER.indx2atm)
    num_propagation_steps = 4 # default value?

    batch_size = 64

    gnn_config = {
        'node_feature_dimension' : node_feature_dimension,
        'num_propagation_steps' : num_propagation_steps
    }
    datamodule_config = {
        'collate_func' : collate_for_graphs,
        'train_batch_size' : batch_size,
        'val_batch_size' : batch_size,
        'num_workers'   :   4
    }

    model = GNN(gnn_config)
    data_module = GNNDataModule(datamodule_config) #config = datamodule_config)
    trainer = pl.Trainer(accelerator='gpu', devices=1)

    trainer.fit(model, data_module)
#    plot_train_and_val_using_mpl(out['train_loss_list'], out['val_lost_list'], name='arbitrary plotname', save=True)

if __name__ == '__main__':
    main()