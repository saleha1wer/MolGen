# perhaps include script for ROC file, etc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import export_text
import torch 
import pandas as pd
from network import GNN
import numpy as np
from utils.from_smiles import from_smiles
from torch_geometric.nn.models import GIN
from data_module import MoleculeDataset,GNNDataModule
import os
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def plot_train_and_val_using_mpl(train_loss, val_loss, name=None, save=False):
    """
    Plots the train and validation loss using Matplotlib
    """
    assert len(train_loss) == len(val_loss)

    f, ax = plt.subplots()
    x = np.arange(len(train_loss))
    ax.plot(x, np.array(train_loss), label='train')
    ax.plot(x, np.array(val_loss), label='val')
    ax.legend()
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    if save:
        if name != None:
            ax.set_title(name)
        else:
            ax.set_title('Anonymous plot')
        plt.savefig(name + '.png')
    return f, ax


def threshold_affinity(values, threshold=6.5):
    new_array = []
    for value in values:
        if value >= threshold:
            new_array.append(1)
        else:
            new_array.append(0)
    return new_array

def plot_roc(y_true,y_preds,labels): 
    """
    y_true is a list of real affinity values:
    [5,6,3,8...]
    y_preds is a list of lists, len > 0. Each list in y_preds being a models predictions and the same size as the true values list:
    [y_pred_model1,y_pred_model2,y_pred_model3...]
    labels are the names of the models in the same order as the y_preds list
    """
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    org_y_true = y_true
    for idx,y_pred in enumerate(y_preds):
        y_true, y_pred = threshold_affinity(org_y_true), threshold_affinity(y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_true,  y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        plt.plot(fpr,tpr,label=labels[idx]+' auc= {}'.format(str(round(auc,3))))
    plt.legend()
    plt.savefig('images/roc_curve')
    plt.show()

def plot_predvreal(y_true,y_pred,name):
    assert(len(y_true)==len(y_pred))
    ax = plt.scatter(y_true, y_pred, c='crimson',s=2)
    # plt.axline([0, 0], [1, 1])
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.ylim(bottom=0, top=12)
    plt.xlim(left=0, right=12)

    plt.savefig('images/true_v_pred_'+name)
    
    plt.show()


df = pd.read_csv('data/a2aar/raw/human_a2aar_ligands',sep=',')
# target_values = df['pchembl_value_Mean'].to_numpy()
# graphs = []  # TUTORIAL METHOD (JOHN BRADSHAW)
# for smiles in df['SMILES']:
#     obj = from_smiles(smiles)
#     graphs.append(obj)


dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',prot_target_encoding=None)
all_train = []
all_test = []

train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.8, random_state=0)
data_train = dataset[train_indices.tolist()]
data_test = dataset[test_indices.tolist()]

datamodule_config = {
    'batch_size': 5600,
    'num_workers': 0
}
data_module = GNNDataModule(datamodule_config, data_train, data_test)
gnn_config = {'N': 5, 'E': 1, 'lr': 0.00011023765672804427, 'hidden': 512, 'layer_type': 
            GIN, 'n_layers': 4, 'pool': 'GlobalAttention', 'batch_size': 64, 'input_heads': 1, 'active_layer': 'last'}

gnn = GNN(gnn_config)
gnn.load_state_dict(torch.load('models/final_GNN'))

trainer = pl.Trainer(max_epochs=50,
                        accelerator='cpu',
                        devices=1,
                        enable_progress_bar=True,
                        enable_checkpointing=True)

all_data_loader = data_module.all_dataloader()

predictions = trainer.predict(gnn, all_data_loader)
sets = all_data_loader.dataset.datasets
target_values = [d.y.numpy()[0][0] for d in sets[0]]
target_values.extend([d.y.numpy()[0][0] for d in sets[1]])
target_values.extend([d.y.numpy()[0][0] for d in sets[2]])
print(len(target_values))
predictions  =predictions[0].numpy().tolist() 
predictions = [i[0] for i in predictions]
print(mean_squared_error(target_values,predictions))

# y_preds = []
# for graph in graphs:
#     y_pred = gnn(graph)
#     y_pred = y_pred.detach().numpy()
#     y_preds.append(y_pred[0][0])

# print(y_preds)
# print(target_values)

plot_predvreal(target_values,predictions,'GNN-finetuned')