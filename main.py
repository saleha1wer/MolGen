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



def main():

    ### LOADING DATAFRAME ###
    try:
        with open('data/ligand_processed.npy', 'rb') as f:
            np = np.load(f)
            df = pd.DataFrame(np)  # should still convert properly
            df.drop('Smiles')

    except:
        print('Could not load "data/ligand_processed.csv" to dataframe')


    ### MAKING TRAIN AND TEST SET (validation?) ###
    df_train, df_test = train_test_split(df, test_size=0.2)
    print(f"df_train shape {df_train.shape}, '\n', df_test shape: {df_test.shape}")



    ### INITIALIZING GRAPH NETWORK ###
    gnn = GNN(len(Graphs.ATOM_FEATURIZER.indx2atm))



    out = train_neural_network(train_dataset=graphs_train, val_dataset=graphs_val, neural_network=gnn, collate_func=collate_for_graphs) #...ETC)


    # plot = plot_train_and_val_using_altair(out['train_loss_list'], out['val_lost_list'])
    # save(plot, 'chart_lr=2e-3.png')  # .pdf doesn't work?

    plot_train_and_val_using_mpl(out['train_loss_list'], out['val_lost_list'])
    plt.savefig('char_lr=2e-3.pdf')



if __name__ == '__main__':
    main()