import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from mol2fingerprint import calc_fps
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from mol2graph import Graphs
from network import GNN, train_neural_network, plot_train_and_val_using_altair


def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)


def main():

    try:

        with open('data/ligand_processed.npy', 'rb') as f:
            print('Loading data')
            target_values = np.load(f)
            fps = np.load(f)
            graphs = np.load(f)
            print(graphs)

    except:

        ## read SMILES
        print('Could not load data. \n Processing data.')


        df = pd.read_csv('DrugEx/data/LIGAND_RAW.tsv', sep='\t', header=0)
        df = df[['Smiles', 'pChEMBL_Value']]
        df = df.dropna(axis=0)  # drop rows with missing values
        # smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
        #     .replace('[N]', 'N').replace('[B]', 'B') \
        #     .replace('[2H]', '[H]').replace('[3H]', '[H]')
        PandasTools.AddMoleculeColumnToFrame(df,'Smiles','Molecule',includeFingerprints=False)
        df['Smiles'] = df['Smiles'].map(canonical_smiles)
        df = df.drop_duplicates(subset=['Smiles'])
        print('Finished Smiles processing')


        # REGRESSION TARGET VALUES
        target_values = df['pChEMBL_Value'].to_numpy()


        # FEATURIZATION OF MOLECULES
        fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEX method)
        print('Finished calculating fingerprints')

        graphs = np.array([[]])  # TUTORIAL METHOD (JOHN BRADSHAW)
        for mol in df['Molecule']:
            graphs = np.append(graphs, Graphs.from_mol(mol))
        print('Finished making graphs')

        # SAVING FEAUTRIZED MOLECULES
        with open('data/ligand_processed.npy', 'wb') as f:
            np.save(f, target_values)
            np.save(f, fps)
            np.save(f, graphs)


    # MAKING TRAIN AND TEST SET (validation?)
    y_train, y_test, fps_train, fps_test, graphs_train, graphs_test = train_test_split(target_values, fps, graphs, test_size=0.2)


    # THIS USES DRUGEX METHOD - FOR BENCHMARKING
    xgb = XGBRegressor()
    xgb.fit(fps_train, y_train)

    y_test_pred = xgb.predict(fps_test)
    print(mean_squared_error(y_test, y_test_pred))


    # GNN METHOD
    gnn = GNN(len(Graphs.ATOM_FEATURIZER.indx2atm))

    # # train_neural_network in network.py script
    # # we need graphs to be a torch.data.Dataloader (?)
    ### out = train_neural_network(train_dataset=graphs_train_dataloader, val_dataset=graphs_val_dataloader #...ETC)

    plot_train_and_val_using_altair(out['train_loss_list'], out['val_lost_list'])
    # print(graphs.shape)
    # print(graphs[0])
    #
    # print(df.describe())
    # print(df.head())


if __name__ == '__main__':
    main()