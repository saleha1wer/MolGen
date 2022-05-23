import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import PandasTools
from utils.mol2fingerprint import calc_fps
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from utils.mol2graph import Graphs
from network import GNN, train_neural_network, plot_train_and_val_using_altair, collate_for_graphs, plot_train_and_val_using_mpl

def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)


def main():
    try:

        df = pd.read_pickle('data/papyrus_ligand.zip')
        df = df[['SMILES', 'pchembl_value_Mean']]
        df = df.dropna(axis=0)
        df['SMILES'] = df['SMILES'].map(canonical_smiles)
        df = df.drop_duplicates(subset=['SMILES'])
        print('Finished preprocessing Smiles')
        PandasTools.AddMoleculeColumnToFrame(df,'SMILES','Molecule',includeFingerprints=False)
        print('Processed Smiles to Mol object')
        target_values = df['pchembl_value_Mean'].to_numpy()
        fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEx method)
        print('Finished calculating fingerprints')
        graphs = np.array([[]])  # TUTORIAL METHOD (JOHN BRADSHAW)
        for mol in df['Molecule']:
            graphs = np.append(graphs, Graphs.from_mol(mol))
        print('Finished making graphs')

    except Exception as e:
        ## read SMILES
        print('Could not load data. \nProcessing data.',e)

        df = pd.read_csv('DrugEx/data/LIGAND_RAW.tsv', sep='\t', header=0)
        df = df[['Smiles', 'pChEMBL_Value']]
        df = df.dropna(axis=0)  # drop rows with missing values
        # smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
        #     .replace('[N]', 'N').replace('[B]', 'B') \
        #     .replace('[2H]', '[H]').replace('[3H]', '[H]')
        df['Smiles'] = df['Smiles'].map(canonical_smiles)
        df = df.drop_duplicates(subset=['Smiles'])
        print('Finished preprocessing Smiles')

        PandasTools.AddMoleculeColumnToFrame(df,'Smiles','Molecule',includeFingerprints=False)
        print('Processed Smiles to Mol object')

        # REGRESSION TARGET VALUES
        target_values = df['pChEMBL_Value'].to_numpy()

        # FEATURIZATION OF MOLECULES
        fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEx method)
        print('Finished calculating fingerprints')

        graphs = np.array([[]])  # TUTORIAL METHOD (JOHN BRADSHAW)
        for mol in df['Molecule']:
            graphs = np.append(graphs, Graphs.from_mol(mol))
        print('Finished making graphs')



    # NOT USED YET
    #
    # # MAKING TRAIN AND TEST SET (validation?)
    # y_train, y_test, fps_train, fps_test, graphs_train, graphs_test = train_test_split(target_values, fps, graphs, test_size=0.2)
    # print(y_train.shape, '\n', fps_train.shape, graphs_train.shape)
    #
    #
    # # GNN METHOD
    # gnn = GNN(len(Graphs.ATOM_FEATURIZER.indx2atm))
    #
    # graphs_train = pd.DataFrame({'x': graphs_train, 'y': y_train})
    # graphs_val = pd.DataFrame({'x': graphs_test, 'y': y_test})
    #
    # out = train_neural_network(train_dataset=graphs_train, val_dataset=graphs_val, neural_network=gnn, collate_func=collate_for_graphs) #...ETC)
    #
    #
    # # plot = plot_train_and_val_using_altair(out['train_loss_list'], out['val_lost_list'])
    # # save(plot, 'chart_lr=2e-3.png')  # .pdf doesn't work?
    #
    # plot_train_and_val_using_mpl(out['train_loss_list'], out['val_lost_list'])
    # plt.savefig('char_lr=2e-3.pdf')
    #


if __name__ == '__main__':
    main()