import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from mol2fingerprint import calc_fp
from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
from mol2graph import Graphs
from network import GNN

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from utils.mol2graph import Graphs
from network import GNN, train_neural_network, plot_train_and_val_using_altair, collate_for_graphs, plot_train_and_val_using_mpl
import pickle5 as pickle
from torch_geometric.data import Data

def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)

def main():
    try:
        # df = pd.read_pickle('data/papyrus_ligand.zip')
        df = pd.read_csv('data/papyrus_ligand')

    except Exception as e:
        ## read SMILES
        print('Could not load data. \nProcessing data: DrugEx/data/LIGAND_RAW.tsv.',e)

        df = pd.read_csv('DrugEx/data/LIGAND_RAW.tsv', sep='\t', header=0)
        df = df[['Smiles', 'pChEMBL_Value']]

    df = df.dropna(axis=0)  # drop rows with missing values
    # smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
    #     .replace('[N]', 'N').replace('[B]', 'B') \
    #     .replace('[2H]', '[H]').replace('[3H]', '[H]')
    df['Smiles'] = df['Smiles'].map(canonical_smiles)
    df = df.drop_duplicates(subset=['Smiles'])
    print('Finished preprocessing Smiles')

    PandasTools.AddMoleculeColumnToFrame(df, 'Smiles', 'Molecule', includeFingerprints=False)
    print('Processed Smiles to Mol object')

    # REGRESSION TARGET VALUES
    target_values = df['pChEMBL_Value'].to_numpy()

    # FEATURIZATION OF MOLECULES
    fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEx method)
    print('Finished calculating fingerprints')

    graphs = np.array([[]])  # TUTORIAL METHOD (JOHN BRADSHAW)
    for mol in df['Molecule']:
        graph = Graphs.from_mol(mol)
        graph_gm_object = Data(x=graph.node_features, edge_index=edge_index, edge_attr=graph_gm_object.edge_features)
        graphs = np.append(graphs, graph_gm_object)
    print('Finished making graphs')

    # data =

    # MAKING TRAIN AND TEST SET (validation?)
    y_train, y_test, fps_train, fps_test, graphs_train, graphs_test = train_test_split(target_values, fps, graphs, test_size=0.2)
    print(y_train.shape, '\n', fps_train.shape, graphs_train.shape)

    # GNN METHOD
    gnn = GNN(len(Graphs.ATOM_FEATURIZER.indx2atm))



if __name__ == '__main__':
    main()