# THIS WORK IS FROM DRUGEX

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors as desc
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit import DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem.GraphDescriptors import BertzCT


def calc_fp(mols, radius=3, bit_len=2048):
    ecfp = calc_ecfp(mols, radius=radius, bit_len=bit_len)
    phch = calc_physchem(mols)
    fps = np.concatenate([ecfp, phch], axis=1)
    return fps


def calc_ecfp(mols, radius=3, bit_len=2048):
    fps = np.zeros((len(mols), bit_len))
    for i, mol in enumerate(mols):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, fps[i, :])
        except:
            pass
    return fps


def calc_physchem(mols):
    prop_list = ['MW', 'logP', 'HBA', 'HBD', 'Rotable', 'Amide',
                 'Bridge', 'Hetero', 'Heavy', 'Spiro', 'FCSP3', 'Ring',
                 'Aliphatic', 'Aromatic', 'Saturated', 'HeteroR', 'TPSA', 'Valence', 'MR']
    fps = np.zeros((len(mols), 19))
    props = Property()
    for i, prop in enumerate(prop_list):
        props.prop = prop
        fps[:, i] = props(mols)
    return fps


class Property:
    def __init__(self, prop='MW'):
        self.prop = prop
        self.prop_dict = {'MW': desc.MolWt,
                          'logP': Crippen.MolLogP,
                          'HBA': AllChem.CalcNumLipinskiHBA,
                          'HBD': AllChem.CalcNumLipinskiHBD,
                          'Rotable': AllChem.CalcNumRotatableBonds,
                          'Amide': AllChem.CalcNumAmideBonds,
                          'Bridge': AllChem.CalcNumBridgeheadAtoms,
                          'Hetero': AllChem.CalcNumHeteroatoms,
                          'Heavy': Lipinski.HeavyAtomCount,
                          'Spiro': AllChem.CalcNumSpiroAtoms,
                          'FCSP3': AllChem.CalcFractionCSP3,
                          'Ring': Lipinski.RingCount,
                          'Aliphatic': AllChem.CalcNumAliphaticRings,
                          'Aromatic': AllChem.CalcNumAromaticRings,
                          'Saturated': AllChem.CalcNumSaturatedRings,
                          'HeteroR': AllChem.CalcNumHeterocycles,
                          'TPSA': AllChem.CalcTPSA,
                          'Valence': desc.NumValenceElectrons,
                          'MR': Crippen.MolMR,
                          'QED': qed,
                          # 'SA': sascorer.calculateScore,  # no working sascorer, left out for now
                          'Bertz': BertzCT}

    def __call__(self, mols):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                scores[i] = self.prop_dict[self.prop](mol)
            except:
                continue
        return scores