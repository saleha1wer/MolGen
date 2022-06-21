import os
from flask import Config
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from network import GNN
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules
from torch_geometric.nn.models import GIN, GAT, PNA, GraphSAGE
from ray.tune.utils import wait_for_gpu
from hpo import run_hpo_basic, run_hpo_finetuning
from finetune import finetune
import torch
from ray import tune
def main():
    pretrain_epochs = 2
    finetune_epochs = 100

from datetime import datetime

def main():
    pretrain_epochs = 100
    finetune_epochs = 50
    adenosine_star = False
    NUM_NODE_FEATURES = 5
    NUM_EDGE_FEATURES = 1
    n_samples = 500  # hpo param
    max_t_per_trial = 2000  # hpo param
    batch_size = 64
    no_a2a = True #use a2a data or not in adenosine set
    train_size = 0.8
    n_inputs = 1
    sec_inp = 'prot' # or 'xgb' or 'fps' 
    include_fps = True if sec_inp == 'fps' else False
    
    #
    ######################################################################################################################
    # HPO on pretrain data (adenosine)
    # best_config = run_hpo(max_epochs=max_epochs, n_samples=n_samples, max_t_per_trial=max_t_per_trial, data_module=pre_data_module, gnn_config=gnn_config)
    # print('BEST CONFIG: ')
    # print(best_config)
    # Pretrain best config on pretrain data
    gnn_config = {
        'N': NUM_NODE_FEATURES,
        'E': NUM_EDGE_FEATURES,
        'lr': tune.loguniform(1e-4, 1e-1),  # learning rate
        'hidden': tune.choice([16, 32, 64, 128, 256, 512]),  # embedding/hidden dimensions
        # 'layer_type': tune.choice([GIN, GAT, GraphSAGE]),
        'layer_type': GIN,
        'n_layers': tune.choice([2, 3, 4, 5, 6, 7]),
        'pool': tune.choice(['mean', 'GlobalAttention']),
        'accelerator': 'cpu',
        'batch_size': batch_size,
        'input_heads': n_inputs,
        'active_layer': tune.choice(['first', 'last']),
        'trade_off_backbone': tune.loguniform(1e-5, 10),
        'trade_off_head': tune.loguniform(1e-5, 1),
        'order': tune.choice([1, 2, 3, 4]),  # is this a good interval?
        'patience': 10,
        'second_input': sec_inp
        # 'batch_size': tune.choice([16,32,64,128])
    }
    pre_data_module, fine_data_module = create_pretraining_finetuning_DataModules(batch_size, no_a2a, train_size, random_state=0, 
                                                                                prot_enc='one-hot-encoding',include_fps=include_fps)

    best_configuration, best_loss = run_hpo_finetuning(pretrain_epochs, finetune_epochs, n_samples, max_t_per_trial, pre_data_module, fine_data_module, gnn_config)

    now_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    file = open('HPO_results.txt','a')
    message = f"\nTime of writing: {now_string}\nLoss achieved: {str(best_loss)} \nConfiguration found: {str(best_configuration)}"
    file.write(message)
    
    # best_configuration =  {'N': 5, 'E': 1, 'lr': 0.00016542323876234363, 'hidden': 256, 
    #         'layer_type': GIN,'n_layers': 6, 'pool': 'mean', 'accelerator': 'cpu', 
    #         'batch_size': 64, 'input_heads': 2, 'active_layer': 'first', 'trade_off_backbone': 0.1,
    #          'trade_off_head': 0.0005, 'order': 1, 'patience': 10, 'second_input': 'fps'}
    
    model = GNN(best_configuration)
    trainer = pl.Trainer(max_epochs=150,
                                accelerator='cpu',
                                devices=1,
                                enable_progress_bar=True)
    trainer.fit(model, pre_data_module)
    torch.save(model.state_dict(), 'models/pretrained_twoinputs_{}.pt',format(best_configuration['second_input']))
    # model.load_state_dict(torch.load('models/final_GNN_two_inputs_fps'))
    # Finetune 
    # Finetune on a2a data
    source_model = model 
    finetuned_model = finetune(save_model_name='final_', source_model=source_model, data_module=fine_data_module,epochs=50,
                            patience=20, trade_off_backbone=best_configuration['trade_off_backbone'],trade_off_head=best_configuration['trade_off_head'],
                            order=best_configuration['order'])
    if n_inputs >1:
        torch.save(finetuned_model.state_dict(),'models/final_GNN_two_inputs_{}'.format(best_configuration['second_input']))
    else:
        torch.save(finetuned_model.state_dict(),'models/final_GNN_one_input')



# def main():
#     best_loss, best_config = meta_hpo_finetuning(pretrain_epochs = 50,
#                                                  finetune_epochs = 30,
#                                                  n_samples = 30,
#                                                  train_size = 0.8)
#     save_loss_and_config(best_loss, best_config)
#     print('Completed a pretrain and finetuning HPO run!')

if __name__ == '__main__':
    main()
