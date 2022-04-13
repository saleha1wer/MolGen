import pandas as pd


def main():
    df = pd.read_csv('DrugEx/data/LIGAND_RAW.tsv', sep='\t', header=0)

    print(df.columns)


    df = df[['Smiles', 'pChEMBL_Value']]
    df = df.dropna(axis=0)  # drop rows with missing values

    PandasTools.AddMoleculeColumnToFrame(df,'Smiles','Molecule',includeFingerprints=False)

    # smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
    #     .replace('[N]', 'N').replace('[B]', 'B') \
    #     .replace('[2H]', '[H]').replace('[3H]', '[H]')

    df = df.drop_duplicates(subset=['Smiles'])


    print(df.describe())
    print(df.head())


if __name__ == '__main__':
    main()