import os
import subprocess
import tempfile




# hidden_sizes = [1,5,10,20,50,100,200,500,1000,2000,4000,5000,7500]

# for size in hidden_sizes:
#     path = 'chemprop_hiddensize/'+str(size)
#     os.mkdir(path)
#     cmd = "chemprop_train --data_path data/a2aar/raw/human_a2aar_ligands --dataset_type regression --smiles_columns SMILES --target_columns pchembl_value_Mean --epochs 35 --hidden_size {} --save_dir {}".format(size,path)
#     os.system(cmd)



extra_hidden_sizes = [350,750,1500,3000,4000,5000,7500]

for size in extra_hidden_sizes:
    path = 'chemprop_hiddensize/'+str(size)
    os.mkdir(path)
    cmd = "chemprop_train --data_path data/a2aar/raw/human_a2aar_ligands --dataset_type regression --smiles_columns SMILES --target_columns pchembl_value_Mean --epochs 35 --hidden_size {} --save_dir {}".format(size,path)
    os.system(cmd)

