import os
import subprocess
import tempfile
import pathlib




hidden_sizes = [1,5,10,20,50,100,200,350,500,600,700,800,900,1000,1250,1500,2000,3000,4000,5000,6000,7500]

for size in hidden_sizes:
    path = 'chemprop_hiddensize/'+str(size)
    if not pathlib.Path(path).exists():
        os.mkdir(path)
        cmd = "chemprop_train --data_path data/a2aar/raw/human_a2aar_ligands --dataset_type regression --smiles_columns SMILES --target_columns pchembl_value_Mean --epochs 35 --hidden_size {} --save_dir {}".format(size,path)
        os.system(cmd)



# extra_hidden_sizes = [10000]

# for size in extra_hidden_sizes:
#     path = 'chemprop_hiddensize/'+str(size)
#     os.mkdir(path)
#     cmd = "chemprop_train --data_path data/a2aar/raw/human_a2aar_ligands --dataset_type regression --smiles_columns SMILES --target_columns pchembl_value_Mean --epochs 35 --hidden_size {} --save_dir {}".format(size,path)
#     os.system(cmd)

