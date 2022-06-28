# perhaps include script for ROC file, etc
from cProfile import label
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import export_text
import torch 
import pandas as pd
from network import GNN
import numpy as np
from utils.encode_ligand import from_smiles
from torch_geometric.nn.models import GIN
from data_module import MoleculeDataset, GNNDataModule,create_pretraining_finetuning_DataModules
import os
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from utils.GINE import GINE, GAT
from scipy.stats import spearmanr
from scipy.stats import gaussian_kde
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import PandasTools
from utils.encode_ligand import calc_fps
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
    # plt.savefig('images/roc_curve')
    plt.show()

def sort_true_and_pred(true, pred):
    zipped_lists = zip(true, pred)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    true, pred = [ list(tuple) for tuple in  tuples]
    return true, pred
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def plot_predvreal(y_true,y_pred,name):
    plt.clf()
    assert(len(y_true)==len(y_pred))
    y_true, y_pred = sort_true_and_pred(y_true,y_pred)
    temp = list(split(y_true,3))
    temp2 = list(split(y_pred,3))
    print('{} preds leingth {}'.format(name,len(y_true)))
    first_y_trues = temp[0]
    first_y_preds = temp2[0]
    second_y_trues = temp[1]
    second_y_preds = temp2[1]
    third_y_trues = temp[2]
    third_y_preds = temp2[2]
    f, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--", c=".3",alpha=0.4)
    ax.scatter(first_y_trues, first_y_preds, c='green',s=2)
    ax.scatter(second_y_trues, second_y_preds, c='grey',s=2)
    ax.scatter(third_y_trues, third_y_preds, c='red',s=2)
    plt.xlabel('True', fontsize=15)
    plt.ylabel('Predicted', fontsize=15)
    plt.ylim(bottom=3, top=11)
    plt.xlim(left=3, right=11)
    plt.savefig('images/true_v_pred_'+name)
    plt.clf()
    # plt.show()

def plot_density(y_true,y_pred,name):
    plt.clf()
    true_density = gaussian_kde(y_true,bw_method=0.5)
    pred_density = gaussian_kde(y_pred,bw_method=0.5)
    x = np.linspace(3,11,500)
    y_true=true_density(x)
    y_pred=pred_density(x)
    plt.figure(figsize=(6,6))
    plt.xlabel('Affinity Value')
    plt.ylabel('Density')
    plt.plot(x, y_pred,label='Predicted',color='darkorange')
    plt.plot(x, y_true, label='True', color='royalblue')
    plt.fill_between(x, y_true, alpha=0.4,color='royalblue')
    plt.fill_between(x, y_pred, alpha=0.4,color='orange')
    plt.legend()
    plt.savefig('images/density_'+name)
    plt.clf()

def get_model(path):
    GAT_params = {'N': 9, 'E':1, 'lr': 0.0001201744224722,'hidden':1024, 'layer_type': GAT , 'n_layers': 7, 
                    'pool': 'GlobalAttention', 'accelerator': 'cpu','dropout_rate':0, 'v2':True, 'batch_size': 64, 
                    'input_heads': 1, 'second_input': None}
    GINE_params =  {'N': 9, 'E':0, 'lr': 0.000013188712926692827,'hidden':512, 'layer_type': GINE , 'n_layers': 8, 
                    'pool': 'GlobalAttention', 'accelerator': 'cpu','dropout_rate':0, 'batch_size': 64, 
                    'input_heads': 1, 'second_input': None}
    if 'xgb-mol' in path:
        GAT_params['input_heads'] = 2
        GAT_params['second_input'] = 'xgb'
        gnn = GNN(GAT_params)
    elif 'prot-mol' in path:
        GAT_params['input_heads'] = 2
        GAT_params['second_input'] = 'prot'
        gnn = GNN(GAT_params)
    elif 'fps-mol' in path:
        GAT_params['input_heads'] = 2
        GAT_params['second_input'] = 'fps'
        gnn = GNN(GAT_params)
    elif 'GIN' in path:
        gnn = GNN(GINE_params)
    elif 'GAT' in path:
        gnn = GNN(GAT_params)
    else: 
        raise ValueError('Model not found')
    return gnn

dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands',prot_target_encoding='one-hot-encoding')
all_train = []
all_test = []

train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.9, random_state=0)
data_train = dataset[train_indices.tolist()]
data_test = dataset[test_indices.tolist()]

datamodule_config = {
    'batch_size': 5600,
    'num_workers': 0
}
data_module = GNNDataModule(datamodule_config, data_train, data_test)
all_data_loader = data_module.all_dataloader()
sets = all_data_loader.dataset.datasets
target_values = [d.y.numpy()[0][0] for d in sets[1]] # test set real values


# aden_dataset = MoleculeDataset(root=os.getcwd() + '/data/adenosine', filename='human_adenosine_ligands',prot_target_encoding='one-hot-encoding')
# all_train = []
# all_test = []

# train_indices, test_indices = train_test_split(np.arange(aden_dataset.len()), train_size=0.9, random_state=0)
# data_train = aden_dataset[train_indices.tolist()]
# data_test = aden_dataset[test_indices.tolist()]
# datamodule_config = {
#     'batch_size': 5600,
#     'num_workers': 0
# }
# aden_data_module = GNNDataModule(datamodule_config, data_train, data_test)
# all_data_loader = aden_data_module.all_dataloader()
# sets = all_data_loader.dataset.datasets
# aden_target_values = [d.y.numpy()[0][0] for d in sets[1]] # test set real values



# folder_path = 'models_saved/'
# for path in ['bestmodel_GAT_trained.pt','bestmodel_GAT_finetuned.pt','bestmodel_GIN_finetuned.pt','bestmodel_GIN_trained.pt','xgb-mol.pt','prot-mol.pt','fps-mol.pt']:
#     if 'prot-mol' not in path:
#         trainer = pl.Trainer(max_epochs=50,
#                             accelerator='cpu',
#                             devices=1,
#                             enable_progress_bar=True,
#                             enable_checkpointing=True)
#         gnn = get_model(folder_path+path)
#         gnn.load_state_dict(torch.load(folder_path+path))
#         predictions = trainer.predict(gnn, data_module.test_dataloader())
#         predictions  =predictions[0].numpy().tolist() 
#         predictions = [i[0] for i in predictions]

#         print('Path: ',path)
#         print('mse: ', mean_squared_error(target_values,predictions))
#         print('CC: ', spearmanr(target_values,predictions))

#         thresh_predictions = threshold_affinity(predictions)
#         thresh_target_values = threshold_affinity(target_values)
#         print('accuracy: ', accuracy_score(thresh_target_values,thresh_predictions))
#         if 'xgb-mol' in path:
#             plot_predvreal(target_values,predictions,'xgb-mol')
#             plot_density(target_values,predictions,'xgb-mol')
#         if 'fps' in path:
#             plot_predvreal(target_values,predictions,'fps-mol')
#             plot_density(target_values,predictions,'fps-mol')
            
#         if 'GAT' in path:
#             if 'finetuned' in path:
#                 plot_predvreal(target_values,predictions,'GAT_finetuned')
#                 plot_density(target_values,predictions,'GAT_finetuned')
#             elif 'trained' in path:
#                 plot_predvreal(target_values,predictions,'GAT_trained')
#                 plot_density(target_values,predictions,'GAT_trained')
                
#             else:
#                 raise ValueError('Model not found')
#         elif 'GIN' in path:
#             if 'finetuned' in path:
#                 plot_predvreal(target_values,predictions,'GIN_finetuned')
#                 plot_density(target_values,predictions,'GIN_finetuned')
#             elif 'trained' in path:
#                 plot_predvreal(target_values,predictions,'GIN_trained')
#                 plot_density(target_values,predictions,'GIN_trained')
#             else:
#                 raise ValueError('Model not found')
#     else:

#         trainer = pl.Trainer(max_epochs=50,
#                             accelerator='cpu',
#                             devices=1,
#                             enable_progress_bar=True,
#                             enable_checkpointing=True)
#         gnn = get_model(folder_path+path)
#         gnn.load_state_dict(torch.load(folder_path+path))
#         predictions = trainer.predict(gnn, aden_data_module.test_dataloader())
#         predictions  =predictions[0].numpy().tolist() 
#         predictions = [i[0] for i in predictions]

#         print('Path: ',path)
#         print('mse: ', mean_squared_error(aden_target_values,predictions))
#         print('CC: ', spearmanr(aden_target_values,predictions))

#         thresh_predictions = threshold_affinity(predictions)
#         thresh_target_values = threshold_affinity(aden_target_values)
#         print('accuracy: ', accuracy_score(thresh_target_values,thresh_predictions))
#         plot_predvreal(aden_target_values,predictions,'prot-mol')
#         plot_density(aden_target_values,predictions,'prot-mol')



##############################################################################################################################

model = XGBRegressor()
model.load_model('models_saved/xgb_models/xgb_a2a.json')

df = pd.read_csv('data/a2aar/raw/human_a2aar_ligands')
df = df[['SMILES', 'pchembl_value_Mean']]
# df = df.dropna(axis=0)
# df['SMILES'] = df['SMILES'].map(canonical_smiles)
PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule', includeFingerprints=False)
print('Processed Smiles to Mol object')
target_values = df['pchembl_value_Mean'].to_numpy()
fps = calc_fps(df['Molecule'])  # FINGERPRINT METHOD (DrugEx method)
print('Finished calculating fingerprints')
print(fps.shape)
print(target_values.shape)

X_train, X_test, y_train, y_test = train_test_split(fps, target_values, test_size=0.1,random_state=0)

predictions = model.predict(X_test)
print('mse: ', mean_squared_error(y_test,predictions))
print('CC: ', spearmanr(y_test,predictions))
thresh_pred = threshold_affinity(predictions)
thresh_real = threshold_affinity(y_test)
print('accuracy: ', accuracy_score(thresh_real,thresh_pred))
plot_predvreal(y_test,predictions,'XGBoost')