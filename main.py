import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from mol2fingerprint import calc_fp
from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
from mol2graph import Graphs
from network import GNN


def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)


def main():
    df = pd.read_csv('DrugEx/data/LIGAND_RAW.tsv', sep='\t', header=0)

    print(df.columns, '\n', df.shape)


    df = df[['Smiles', 'pChEMBL_Value']]
    df = df.dropna(axis=0)  # drop rows with missing values

    # smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
    #     .replace('[N]', 'N').replace('[B]', 'B') \
    #     .replace('[2H]', '[H]').replace('[3H]', '[H]')

    PandasTools.AddMoleculeColumnToFrame(df,'Smiles','Molecule',includeFingerprints=False)

    df['Smiles'] = df['Smiles'].map(canonical_smiles)

    df = df.drop_duplicates(subset=['Smiles'])

    df_train, df_test = train_test_split(df, test_size=0.2)

    print(df_train.shape)

    # fps = calc_fp(df['Molecule'])


    # data = np.array([fps, df['pChEMBL_Value']])
    #
    # train =

    graphs = np.array([[]])
    for mol in df['Molecule']:
        graphs = np.append(graphs, Graphs.from_mol(mol))

    gnn = GNN(len(Graphs.ATOM_FEATURIZER.indx2atm))




    print(graphs.shape)
    print(graphs[0])

    print(df.describe())
    print(df.head())


if __name__ == '__main__':
    main()