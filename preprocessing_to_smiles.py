# THIS WORK IS FROM DRUGEX
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
from DrugEx.utils import VocSmiles, VocGraph  # utils.vocab (?)
import argparse
from DrugEx import utils
import re
import numpy as np
from itertools import combinations
import gzip
import getopt, sys
rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.warning')



def load_molecules(base_dir, input_file):
    """
    Loads raw SMILES from input file and transform to rdkit molecule
    Arguments:
        base_dir (str)            : base directory, needs to contain a folder data with input file
        input_file  (str)         : file containing SMILES, can be 'sdf.gz' or (compressed) 'tsv' or 'csv' file
    Returns:
        mols (lst)                : list of rdkit-molecules
    """

    print('Loading molecules...')

    file_path = base_dir + '/data/' + input_file

    if input_file.endswith('sdf.gz'):
        # read molecules from file
        inf = gzip.open(file_path)
        mols = Chem.ForwardSDMolSupplier(inf)

    else:
        # read molecules from file and drop duplicate SMILES
        df = pd.read_csv(file_path, sep='\t' if '.tsv' in input_file else ',',
                         usecols= lambda x : x.upper() in ['SMILES'])
        # df.columns.str.upper()
        df = df.SMILES.dropna().drop_duplicates()
        mols = [Chem.MolFromSmiles(s) for s in df]

    return mols

def standardize_mol(mols):
    """
    Standardizes SMILES and removes fragments
    Arguments:
        mols (lst)                : list of rdkit-molecules
    Returns:
        smiles (set)              : set of SMILES
    """

    print('Standardizing molecules...')

    charger = rdMolStandardize.Uncharger()
    chooser = rdMolStandardize.LargestFragmentChooser()
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    smiles = set()
    carbon = Chem.MolFromSmarts('[#6]')
    salts = Chem.MolFromSmarts('[Na,Zn]')
    for mol in tqdm(mols):
        try:
            mol = disconnector.Disconnect(mol)
            mol = normalizer.normalize(mol)
            mol = chooser.choose(mol)
            mol = charger.uncharge(mol)
            mol = disconnector.Disconnect(mol)
            mol = normalizer.normalize(mol)
            smileR = Chem.MolToSmiles(mol, 0)
            # remove SMILES that do not contain carbon
            if len(mol.GetSubstructMatches(carbon)) == 0:
                continue
            # remove SMILES that still contain salts
            if len(mol.GetSubstructMatches(salts)) > 0:
                continue
            smiles.add(Chem.CanonSmiles(smileR))
        except:
            print('Parsing Error:', Chem.MolToSmiles(mol))

    return smiles


def corpus(base_dir, smiles, output, voc_file, save_voc=True):
    """
    Tokenizes SMILES and returns corpus of tokenized SMILES and vocabulary of all the unique tokens
    Arguments:
        base_dir (str)            : base directory, needs to contain a folder data with .tsv file containing dataset
        smiles  (str)             : list of standardized SMILES
        output (str)              : name of output corpus file
        voc_file (str)            : name of output voc_file
        save_voc (bool)           : if true save voc file (should only be true for the pre-training set)
    """

    print('Creating the corpus...')
    voc = VocSmiles()
    # set of unique tokens
    words = set()
    # original SMILES
    canons = []
    # tokenized SMILES
    tokens = []
    for smile in tqdm(smiles):
        if len(smile) <= 100 or len(smile) > 10:
        token = voc.split(smile)
        # keep SMILES within certain length
        if 10 < len(token) <= 100:
            words.update(token)
            canons.append(smile)
            tokens.append(' '.join(token))

    # save voc file
    if save_voc:
        print('Saving vocabulary...')
        log = open(base_dir + '/data/%s_smiles.txt' % voc_file, 'w')
        log.write('\n'.join(sorted(words)))
        log.close()

    log = pd.DataFrame()
    log['Smiles'] = canons
    log['Token'] = tokens
    log.drop_duplicates(subset='Smiles')
    log.to_csv(base_dir + '/data/' + output + '_corpus.txt', sep='\t', index=False)


def DatasetArgParser(txt=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file containing raw data. tsv or sdf.gz format")
    parser.add_argument('-o', '--output', type=str, default='ligand',
                        help="Prefix of output files")
    parser.add_argument('-mt', '--mol_type', type=str, default='smiles',
                        help="Type of molecular representation: 'graph' or 'smiles'")
    parser.add_argument('-nof', '--no_frags', action='store_true',
                        help="If on, molecules are not split to fragments and a corpus is create")

    parser.add_argument('-fm', '--frag_method', type=str, default='brics',
                        help="Fragmentation method: 'brics' or 'recap'")
    parser.add_argument('-nf', '--n_frags', type=int, default=4,
                        help="Number of largest leaf-fragments used per compound")
    parser.add_argument('-nc', '--n_combs', type=int, default=None,
                        help="Maximum number of leaf-fragments that are combined for each fragment-combinations. If None, default is {n_frags}")

    parser.add_argument('-vf', '--voc_file', type=str, default='voc',
                        help="Name for voc file, used to save voc tokens")
    parser.add_argument('-sv', '--save_voc', action='store_true',
                        help="If on, save voc file (should only be done for the pretraining set). Currently only works is --mol_type is 'smiles'.")
    parser.add_argument('-sif', '--save_intermediate_files', action='store_true',
                        help="If on, intermediate files")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")

    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()

    if args.n_combs is None:
        args.n_combs = args.n_frags

    if args.no_git is False:
        args.git_commit = utils.commit_hash(os.path.dirname(os.path.realpath(__file__)))
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(args.base_dir + '/data_args.json', 'w') as f:
        json.dump(vars(args), f)

    print('completed parser')

    return args


def Dataset(base_dir, input_file):
    """
    Prepare input files for DrugEx generators containing encoded molecules for three different cases:

    - SMILES w/o fragments: {output}_corpus.txt and [opt] {voc}_smiles.txt containing the SMILES-token-encoded molecules and the token-vocabulary respectively
    - SMILES w/ fragments: {output}_{mf/sf}_{frag_method}_[train/test]_smi.txt and [opt] {voc}_smiles.txt containing the SMILES-token-encoded fragment-molecule pairs for the train and test sets and the token-vocabulary respectively
    - Graph fragments: {output}_{mf/sf}_{frag_method}_[train/test]_graph.txt and [opt] {voc}_graph.txt containing the encoded graph-matrices of fragement-molecule pairs for the train and test sets and the token-vocabulary respectively
    """

    # load molecules
    mols = load_molecules(args.base_dir, args.input)
    # standardize smiles and remove salts
    smiles = standardize_mol(mols)

    if args.no_frags:
        if args.mol_type == 'graph':
            raise ValueError("To apply --no_frags, --mol_type needs to be 'smiles'")
        # create corpus (only used in v2), vocab (only used in v2)
        corpus(args.base_dir, smiles, args.output, args.voc_file, args.save_voc)


if __name__ == '__main__':
    # args = DatasetArgParser('-b DrugEx/data')
    parser = argparse()
    Dataset(base_dir='DrugEx', input_file='LIGAND_RAW.tsv')