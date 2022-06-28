import os
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules
from torch_geometric.nn.models import GIN
from hpo import run_hpo_basic
from finetune import finetune
import torch
from ray import tune
from network import GNN
from utils.GINE import GINE,GAT
import argparse


def main(mode,type,epochs,folder_path,file_name,save_dir,n_samples,pretrain_path):
    input_heads = 1
    second_input = None
    batch_size = 64
    filename = 'trained_model'
    gnn = type
    GAT_params = {'N': 9, 'E':1, 'lr': 0.0001201744224722,'hidden':1024, 'layer_type': GAT , 'n_layers': 7, 
                        'pool': 'GlobalAttention', 'accelerator': 'cpu','dropout_rate':0, 'v2':True, 'batch_size': 64, 
                        'input_heads': input_heads, 'second_input': second_input}
    GINE_params =  {'N': 9, 'E':0, 'lr': 0.000013188712926692827,'hidden':512, 'layer_type': GINE , 'n_layers': 8, 
                        'pool': 'GlobalAttention', 'accelerator': 'cpu','dropout_rate':0, 'batch_size': 64, 
                        'input_heads': input_heads, 'second_input': second_input}
    finetune_params = {'order': 10,  'trade_off_head': 0.021585620269150924, 'trade_off_backbone': 0.0016808005879414354}  
    # Create the data module
    dataset = MoleculeDataset(root=os.getcwd() + folder_path, filename=file_name,
                                prot_target_encoding=None,xgb=None,include_fps=False)
    train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.9, random_state=0)
    data_train = dataset[train_indices.tolist()]
    data_test = dataset[test_indices.tolist()]
    datamodule_config = {
    'batch_size': batch_size,
    'num_workers': 0
    }
    data_module = GNNDataModule(datamodule_config,data_train,data_test)
    if mode == 'train':
        print('Starting Training')
        params = GAT_params if gnn == 'GAT' else GINE_params
        print('Model Params: ')
        print(params)
        model = GNN(params)
        # Create the trainer
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=epochs)
        # Train the model
        trainer.fit(model, data_module)
        # Save the model
        torch.save(model.state_dict(), './{}/{}.pt'.format(save_dir,filename))
        print('Training Completed, saved model --> /{}/{}.pt'.format(save_dir,filename))
    elif mode == 'hpo':
        GAT_space = {'N': 9, 'E':1, 'lr':tune.loguniform(1e-4, 1e-1),'hidden': tune.choice([16, 32, 64, 128, 256, 512,1024]), 
                'layer_type': GAT , 'n_layers': tune.choice([2, 3, 4, 5, 6, 7]), 'pool': 'GlobalAttention', 'accelerator': 'cpu','dropout_rate':tune.uniform(0,0.5), 
                'v2':True, 'batch_size': 64, 'input_heads': input_heads, 'second_input': second_input}
        GINE_space = {'N': 9, 'E':0, 'lr':tune.loguniform(1e-4, 1e-1),'hidden': tune.choice([16, 32, 64, 128, 256, 512,1024]), 
                'layer_type': GINE , 'n_layers': tune.choice([2, 3, 4, 5, 6, 7]), 'pool': 'GlobalAttention', 'accelerator': 'cpu','dropout_rate':tune.uniform(0,0.5), 
                'v2':True, 'batch_size': 64, 'input_heads': input_heads, 'second_input': second_input}
        space = GAT_space if gnn == 'GAT' else GINE_space
        run_hpo_basic(epochs,n_samples,data_module,space)
    elif mode == 'finetune':
        source_config = GAT_params if gnn == 'GAT' else GINE_params
        # best hpo finetune params are:
        finetune_params = {'order': 10,  'trade_off_head': 0.021585620269150924, 'trade_off_backbone': 0.0016808005879414354}   
        source_model = GNN(source_config)
        if gnn == 'GAT':
            source_model.load_state_dict(torch.load(pretrained_model_path))
        elif gnn == 'GINE':
            source_model.load_state_dict(torch.load(pretrained_model_path))
        print('Finetuning Model')
        finetuned_model, train_losses,val_losses = finetune('Finetuned_{}'.format(gnn),source_model,data_module,epochs=epochs,report_to_raytune=False,patience=30,
                                                            order = finetune_params['order'],trade_off_backbone=finetune_params['trade_off_backbone'],
                                                            trade_off_head=finetune_params['trade_off_head'],fname='finetune_logs_{}'.format(gnn))
        torch.save(finetuned_model.state_dict(), './{}/{}.pt'.format(save_dir,filename))
        print('Finetuning Completed, saved model --> /{}/{}.pt'.format(save_dir,filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GenMol Affinity Predictor')
    parser.add_argument('mode', type=str,
                    help='train, hpo, or finetune')
    parser.add_argument('type', type=str,
                    help='GIN or GAT')
    parser.add_argument('--epochs', type=int,
                    help='Number of epochs to train. If HPO: number of epochs per sample. Default: 50')
    parser.add_argument('--save-dir', type=str,
                    help='Directory to save the trained model. Default: `models_saved`')
    parser.add_argument('--folder_path', type=str,
                    help='Path to folder with csv file with SMILES and target values. Default: `/data/a2aar`')
    parser.add_argument('--file_name', type=str,
                    help='Path to csv file with SMILES and target values. Default: `human_a2aar_ligands`')
    parser.add_argument('--n_samples', type=int,
                    help='If mode is hpo: number of samples to run. Default: 30')
    parser.add_argument('--pretrained_model_path', type=str,
                    help='If mode is fintune, path to pretrained model. Default: `./models_saved/<type>_pretrained.pt`')
    args = parser.parse_args()
    # main(mode='train')
    mode = args.mode
    type = args.type
    epochs = args.epochs
    save_dir = args.save_dir
    folder_path = args.folder_path
    file_name = args.file_name
    n_samples = args.n_samples
    pretrained_model_path = args.pretrained_model_path

    if save_dir == None:
        save_dir = 'models_saved'
    if folder_path == None: 
        folder_path = '/data/a2aar'
    if file_name == None:
        file_name = 'human_a2aar_ligands'
    if epochs == None:
        epochs = 50
    if n_samples == None:
        n_samples = 30
    if pretrained_model_path == None:
        pretrained_model_path = './models_saved/GAT_pretrained.pt' if type == 'GAT' else './models_saved/GIN_pretrained.pt'
    main(mode=mode, type=type, epochs=epochs,folder_path=folder_path, file_name=file_name, save_dir=save_dir,
         n_samples=n_samples, pretrain_path=pretrained_model_path)
