
import pandas as pd

path = 'data/Papyrus.tsv.gz'

df = pd.read_csv(path, sep='\t')
df = df[['SMILES','Activity_class','target_id','pchembl_value','relation']]

df = df[['Smiles','Molecule_ChEMBL_ID','pChEMBL_Value']]
rec_ID = 'P29274'
# Only keep molecules with the target_ID for the receptor we want
df = df[df['target_id'].str.contains(rec_ID)]

print(df.head())
print('Length: ', df.shape[0])
df.to_pickle('papryus_ligand.zip')



