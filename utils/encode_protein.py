import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ADENOSINE_PROTEIN_IDS = {'P29274_WT': 0, 'P30542_WT': 1, 'P0DMS8_WT': 2, 'P29275_WT': 3}

def get_amino_number(letter):
    letter_to_number = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10,
                        'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'O': 20,
                        'U': 21}
    return letter_to_number[letter]


def prot_target2array(target_id):
    df = pd.read_csv('data/protein_targets.gz', sep='\t')
    df = df[df['target_id'].str.contains(target_id)]
    sequence = str(df['Sequence'].values[0])
    prot_array = []
    for l in sequence:
        idx = get_amino_number(l)
        col = np.zeros(22).tolist()
        col[idx] = 1
        prot_array.append(col)

    for i in range(764 - len(prot_array)):
        col = np.zeros(22).tolist()
        prot_array.append(col)
    prot_array = np.array(prot_array)

    return prot_array

def prot_target2one_hot(target_id):
    category = torch.tensor(ADENOSINE_PROTEIN_IDS[target_id])
    return F.one_hot(category, 4).to(torch.float).view(1, 4).detach().clone()


def prot_target2graph(target_id):
    pass
