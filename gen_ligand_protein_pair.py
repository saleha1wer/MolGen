import pandas as pd

ligand_path = 'data/Papyrus.tsv.gz'

proteins_path = 'data/protein_targets.gz'

raw_ligands = pd.read_csv(ligand_path, sep='\t')
raw_proteins = pd.read_csv(proteins_path, sep='\t')

# todo: Remove low quality data from raw ligands

# Make a dataframe with SMILES, pchembl_value_Mean, target_Id, UniProtID, Classification, Length, Sequence
