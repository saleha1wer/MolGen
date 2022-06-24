import os
from flask import Config
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules
from torch_geometric.nn.models import GAT, GIN
from ray.tune.utils import wait_for_gpu
from hpo import run_hpo_basic, run_hpo_finetuning, meta_hpo_finetuning, save_loss_and_config, calculate_test_loss, \
    meta_hpo_basic
from finetune import finetune
import torch
from ray import tune
from network import GNN
from finetune import finetune

def main(hpo_ft):
    if hpo_ft:
        finetune_epochs = 50
        patience = 15
        n_samples = 30
        source_config = {'N': 9, 'E':1, 'lr': 0.0001201744224722,'hidden':1024, 'layer_type': GAT , 'n_layers': 7, 
                        'pool': 'GlobalAttention', 'accelerator': 'cpu','dropout_rate':0, 'v2':True, 'batch_size': 64, 
                        'input_heads': 1, 'active_layer': 'last', 'second_input': None}
        space = {'order':tune.choice([1,2,3]), 'trade_off_head': tune.choice([tune.loguniform(1e-4, 1e-1),tune.uniform(0.5, 1)]),
                 'trade_off_backbone':tune.choice([tune.loguniform(5e-6, 1e-1),tune.uniform(0.5, 1)])}
        
        predatamodule, finedatamodule = create_pretraining_finetuning_DataModules(64, True, 0.9)
        source_model = GNN(source_config)
        # source_trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=150)
        # source_trainer.fit(source_model, predatamodule.train_dataloader())
        # source_trainer.test(source_model, predatamodule.test_dataloader())
        # torch.save(source_model.state_dict(), 'models_saved/bestconfig_GAT_pretrained')
        
        source_model.load_state_dict(torch.load('models_saved/bestconfig_GAT_pretrained'))
        # best_val_loss, best_configuration = meta_hpo_finetuning(finetune_epochs, patience, n_samples, 0.9,source_model, space)
        finetuned_model, train_losses,val_losses = finetune('test',source_model,finedatamodule,30,True,10,trade_off_backbone=2.5,trade_off_head=0.0005)
        torch.save(finetuned_model.state_dict(), 'models_saved/bestconfig_GAT_finetuned')
        print(train_losses)
        print(val_losses)
        # print(best_val_loss, best_configuration)
    else:
        pretrain_epochs = 50
        train_size = 0.9
        no_a2a = True
        batch_size = 64
        n_inputs = 1
        second_input = 'fps'
        space = {'N': 9, 'E': 1, 'lr': tune.loguniform(1e-5, 1e-2), 'hidden': tune.choice([64, 128, 256, 512, 1024]),
                'layer_type': GIN, 'n_layers': tune.choice([2,4,6,8]), 'pool': tune.choice(['mean', 'GlobalAttention']),
                'dropout_rate' : tune.choice([0,0.1,0.3,0.5]), 'accelerator': 'cpu', 'batch_size': 64,
                'input_heads': 1, 'active_layer': 'last', 'second_input': None}
        
        best_val_loss,best_test_loss, best_config = meta_hpo_basic(pretrain_epochs,
                                                    n_samples = 50,
                                                    train_size = train_size,
                                                    space = space,
                                                    report_test_loss = True)
        print('best config')
        print(best_config)
        save_loss_and_config(best_val_loss,best_test_loss, best_config)
        print('Completed a basic HPO run!')


if __name__ == '__main__':
    main(hpo_ft=True)

