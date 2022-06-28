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

## Installation
* Python >= 3.7.4 
```
pip install -r requirements.txt
```
For CUDA installation
```
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
pip install rdkit pytorch-lightning ray[tune] matplotlib setuptools==59.5 optuna hpbandster ConfigSpace
```
## Usage
From the root directory run: 
```
python main.py <mode> <type> 
```
### Positional arguments:
  * mode:                  train, hpo, or finetune <br/>
  * type:                  GIN or GAT <br/>
  
Train mode with GIN or GAT type will initiate a new model with best parameters
- GIN: 512 Hidden Size, 7 layers, 0 dropout... 
- GAT: 1024 Hidden Size, 8 layers, 0 dropout <br/>

The model is trained on the dataset (default is A2AAR) which should be a csv file with SMILES and p_chembl_value columns. The model is then saved in --save_dir.

HPO mode will run TPE with the specified model type for --n_samples 

Finetune mode will finetune a pretrained model on a dataset (default is A2AAR)

### Optional arguments:
<pre>
  --epochs                Number of epochs to train. If hpo, number of epochs per sample. 
                          Default: 50
                          
  --save-dir              Directory to save the trained model. 
                          Default: models_saved
                          
  --folder_path           Path to folder with csv file with SMILES and target values. 
                          Default: /data/a2aar
                         
  --file_name             Name of file with SMILES and target values. 
                          Default: human_a2aar_ligands
                          
  --n_samples             If mode is hpo: number of samples to run. 
                          Default: 30
                          
  --pretrained_model_path If mode is finetune, path to pretrained model. 
                          Default: ./models_saved/*type*_pretrained.pt
<pre>
