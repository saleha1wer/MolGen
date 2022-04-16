from email.utils import encode_rfc2231
import imp
from os import remove
import pandas as pd 
import rdkit.Chem as Chem
import numpy as np
import math


# for line in file 
# get smiles str 
# get molecule from smiles (rdkit.chem.MolFromSmiles)
# encode node features 
# encod
# e edge list and edge features
#  
class SymbolFeaturizer:
    #https://github.com/john-bradshaw/ml-in-bioinformatics-summer-school-2020/blob/master/ML_for_Molecules_solutions.ipynb
    """
    Symbol featurizer takes in a symbol and returns an array representing its
    one-hot encoding.
    """
    def __init__(self, symbols, feature_size=None):
        self.atm2indx = {k:i for i, k in enumerate(symbols)}
        self.indx2atm = {v:k for k,v in self.atm2indx.items()}
        self.feature_size = feature_size if feature_size is not None else len(symbols)
    
    def __call__(self, atom_symbol):
        out = np.zeros(self.feature_size)
        out[self.atm2indx[atom_symbol]] = 1.
        return out


def remove_nan(mols,af_values):
    nan_idx = []
    for idx,value in enumerate(af_values):
        if math.isnan(value):
            nan_idx.append(idx)

    mols = np.delete(mols,nan_idx)
    af_values = np.delete(af_values,nan_idx)
    return mols,af_values

def read_mols_values(dataframe):
    
    mols = np.array(dataframe["Smiles"])
    af_values = np.array(dataframe['pChEMBL_Value'])

    nan_idx = []
    for idx,mol in enumerate(mols):
        if not isinstance(mol, str):
            nan_idx.append(idx)

    mols = np.delete(mols,nan_idx)
    af_values = np.delete(af_values,nan_idx)


    mols = np.array([Chem.MolFromSmiles(mol) for mol in mols])
    return mols,af_values


def mol_to_edge_list_graph(mol, atm_featurizer):
    # https://github.com/john-bradshaw/ml-in-bioinformatics-summer-school-2020/blob/master/ML_for_Molecules_solutions.ipynb
    # Node features
    node_features = [atm_featurizer(atm.GetSymbol()) for atm in mol.GetAtoms()]
    node_features = np.array(node_features, dtype=np.float32)

    # Edge list and edge feature list
    edge_list = []
    edge_feature_list = []
    for bnd in mol.GetBonds():
        bnd_indices = [bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()]
        bnd_type = bnd.GetBondTypeAsDouble()
        edge_list.extend([bnd_indices,  bnd_indices[::-1]])
        edge_feature_list.extend([bnd_type, bnd_type])
    edge_list = np.array(edge_list, dtype=np.int32)
    edge_feature_list = np.array(edge_feature_list, dtype=np.float32)
    return node_features, edge_list, edge_feature_list
    

    

def smiles_to_graph(smiles,values,atm_featurizer):
    # smiles is a list of mols in smiles representation 
    # returns array of (node_features, edge_list, edge_features,af_value) rows
    X = []
    y = []
    for idx,mol in enumerate(smiles):
        try: 
            node_features, edge_list, edge_feature_list = mol_to_edge_list_graph(mol,atm_featurizer)
            temp = [node_features, edge_list, edge_feature_list]
            X.append(temp)
            y.append(values[idx])
        except Exception as e:
            print(e)

        
    return np.array(X,dtype=object), np.array(y,dtype=object)



def encode_ligands(path,atm_featurizer):
    # Load raw ligands
    df = pd.read_csv(path, sep='\t')
    # Get smiles strings and affinity values
    mols, af_values = read_mols_values(df) 
    # Remove ligands with affinity nan
    mols, af_values = remove_nan(mols,af_values)
    print(len(mols), 'ligands read after removing nans') 
    # Make X --> (node_features, edge_list, edge_features) and y --> af_values arrays
    X,y = smiles_to_graph(mols, af_values,atm_featurizer)
    np.save('data/ligand_processed',X)
    np.save('data/affinity_values',y)



if __name__ == '__main__':
    path = 'data/LIGAND_RAW.tsv'
    voc_path = 'data/ligand_voc.txt'
    vocab = pd.read_csv(voc_path, sep='\t')
    vocab = np.array(vocab).ravel()
    atm_featurizer = SymbolFeaturizer(vocab)
    encode_ligands(path,atm_featurizer)
    
    # print(np.load('affinity_values.npy',allow_pickle=True))
    # print(np.load('ligand_processed.npy',allow_pickle=True))