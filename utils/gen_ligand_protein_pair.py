
import pandas as pd




def get_sequence(protein_ID,file_path):
    """ 
    Returns sequence of protein given ID
    """
    proteins = pd.read_csv(file_path, sep='\t')

    df = proteins[proteins['target_id'].str.contains(protein_ID)]

    if df.shape[0] > 1:
        raise ValueError('More than one protein with the same ID:', protein_ID)
    elif df.shape[0] == 0:
        raise ValueError('No protein ID:', protein_ID)

    return str(df['Sequence'].iloc[0])

