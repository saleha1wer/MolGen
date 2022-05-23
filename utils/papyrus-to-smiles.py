
import pandas as pd

path = 'data/Papyrus.tsv.gz'

df = pd.read_csv(path, sep='\t')
print(df.head())

df = df[['SMILES','Activity_class', 'target_id', 'pchembl_value_Mean', 'relation']]
# df = df[['SMILES','Molecule_ChEMBL_ID','pchembl_value_mean']]
rec_ID = 'P29274'
# Only keep molecules with the target_ID for the receptor we want
df = df[df['target_id'].str.contains(rec_ID)]

print(df.head())
print('Length: ', df.shape[0])
df.to_pickle('papyrus_ligand.zip')


# P29274
# P30542
# P29275
# P0DMS8
