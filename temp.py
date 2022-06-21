from network import GNN
from torch_geometric.nn import GIN
import torch
from data_module import GNNDataModule, MoleculeDataset, create_pretraining_finetuning_DataModules
import pytorch_lightning as pl


config =  {'N': 5, 'E': 1, 'lr': 0.00016542323876234363, 'hidden': 256, 
        'layer_type': GIN,'n_layers': 6, 'pool': 'mean', 'accelerator': 'cpu', 
        'batch_size': 64, 'input_heads': 2, 'active_layer': 'first', 'trade_off_backbone': 8.141935107421304e-05,
            'trade_off_head': 0.12425374868175541, 'order': 1, 'patience': 10,'second_input':'prot'}
model = GNN(config)

no_a2a = True #use a2a data or not in adenosine set
no_a2a = '_no_a2a' if no_a2a else ''
pre_data_module, fine_data_module = create_pretraining_finetuning_DataModules(64, no_a2a, 0.1,prot_enc='one-hot-encoding')
trainer = pl.Trainer(max_epochs=150,
                            accelerator='cpu',
                            devices=1,
                            enable_progress_bar=True)
trainer.fit(model, fine_data_module)
# torch.save(model.state_dict(), 'models/pretrained_best_config.pt')