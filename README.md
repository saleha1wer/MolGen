# Predicting Drug-Target Binding Affinity with PyTorch Geometric

## Data Folder
Contains the following folders:
- a2aar :   A2AAR dataset
- adenosine :  ADENOSINE dataset
- adenosine_no_a2aar : ADENOSINE with A2AAR removed dataset


## utils Folder
Contains the following files:
- encode_ligands.py : change ligand to PyG Data or vector form (fingerprint)
- encode_protein.py : change protein to one-hot-encoding, 
- papyrus-to-smiles : select ligand and protein


Files
- baseline.py : XGBoost baseline
- data_module.py : contains data_module for data processing and saving and contains GNNDataModule
- evaluation.py : visualization of results
- finetune.py : finetunes the final layer with model ensemble regularization
- hpo.py : RayTune optimization setup - callbacks, search algorithm, scheduler, and testing
- main.py : runs HPO
- network.py : contains GNN as pl.LightningModule


# Installation
For CUDA installation
- conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
- conda install pyg -c pyg -c conda-forge
- pip install rdkit pytorch-lightning ray[tune] matplotlib setuptools==59.5 optuna hpbandster ConfigSpace