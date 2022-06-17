from ray import tune
from torch_geometric.nn.models import GIN, GAT,GraphSAGE
from network import GNN
import torch 
import pytorch_lightning as pl
from data_module import GNNDataModule, MoleculeDataset
from sklearn.model_selection import train_test_split
import os 
import numpy as np
from torch_geometric.data import DataLoader
import torch.optim as optim
from GTOT_Tuning.chem.ftlib.finetune.delta import IntermediateLayerGetter, L2Regularization, get_attribute, SPRegularization, FrobeniusRegularization
from GTOT_Tuning.chem.ftlib.finetune.gtot_tuning import GTOTRegularization
from torch import nn
from GTOT_Tuning.chem.commom.early_stop import EarlyStopping
from GTOT_Tuning.chem.commom.run_time import Runtime
from GTOT_Tuning.chem.commom.eval import Meter
from GTOT_Tuning.chem.splitters import scaffold_split, random_split, random_scaffold_split
from tensorboardX import SummaryWriter
from tqdm import tqdm

criterion = nn.BCEWithLogitsLoss(reduction="none")
def train_epoch(model, device, loader, optimizer, weights_regularization, backbone_regularization,
                head_regularization, target_getter,
                source_getter, bss_regularization,trade_off_backbone, trade_off_head, scheduler, epoch):
    model.train()
    meter = Meter()
    loss_epoch = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):
        batch = batch.to(device)
        intermediate_output_s, output_s = source_getter(batch.x, batch.edge_index, batch.edge_attr,
                                                        batch.batch)  # batch.batch is a column vector which maps each node to its respective graph in the batch
        intermediate_output_t, output_t = target_getter(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = output_t
        fea_s = source_getter._model.get_bottleneck()
        fea = target_getter._model.get_bottleneck()
        # intermediate_output_s
        y = batch.y.view(pred.shape).to(torch.float64)
        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        cls_loss = torch.sum(loss_mat) / torch.sum(is_valid)
        meter.update(pred, y, mask=is_valid)
        loss_reg_head = head_regularization()
        loss_reg_backbone = 0.0
        print_str = ''
        loss = torch.tensor([0.0], device=device)
        loss_bss = 0.0
        loss_weights = torch.tensor([0.0]).to(cls_loss.device)
        if trade_off_backbone > 0.0:
            loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t, batch)
        else:
            loss_reg_backbone = backbone_regularization()
        loss = loss + cls_loss + trade_off_backbone * loss_reg_backbone + trade_off_head * loss_reg_head 
        loss = loss + 0.1 * loss_weights
        # if torch.isnan(cls_loss):  # or torch.isnan(loss_reg_backbone):
        #     print(pred, loss_reg_backbone)
        #     raise
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
        optimizer.step()
        loss_epoch.append(cls_loss.item())
    avg_loss = sum(loss_epoch) / len(loss_epoch)
    if scheduler is not None: scheduler.step()
    try:
        print('num_oversmooth:', backbone_regularization.num_oversmooth, end=' || ')
        backbone_regularization.num_oversmooth = 0
    except:
        pass
    metric = np.mean(meter.compute_metric('roc_auc_score_finetune'))
    return metric, avg_loss

dataset = MoleculeDataset(root=os.getcwd() + '/data/a2aar', filename='human_a2aar_ligands')

train_indices, test_indices = train_test_split(np.arange(dataset.len()), train_size=0.8, random_state=0)
data_train = dataset[train_indices.tolist()]
data_test = dataset[test_indices.tolist()]

datamodule_config = {
    'batch_size': 64,
    'num_workers': 0}
data_module = GNNDataModule(datamodule_config, data_train, data_test)
trainer = pl.Trainer(max_epochs=100,
                        accelerator='cpu',
                        devices=1,
                        enable_progress_bar=True)

test_data_loader = data_module.test_dataloader()


## Finetuning
order = 1 
epochs = 30
tag = 'gtot_cosine'
patience=20

trade_off_backbone =  1  # between 5e-6 and 1
trade_off_head = 0.0005  # between 5e-6 and 1
# fine tuning params^  - can different learning rate for each part of the gnn

device = torch.device("cuda:" + str(1)) if torch.cuda.is_available() else torch.device("cpu")



train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
gnn_config = {
    'N': 9,
    'E': 1,
    'lr': 0.002,  # learning rate
    'hidden': 64,  # embedding/hidden dimensions
    'layer_type': GIN,
    'n_layers': 3
    # 'batch_size': tune.choice([16,32,64,128])
}
finetuned_model = GNN(gnn_config)
source_model = GNN(gnn_config)

finetuned_model.to(device)
source_model.to(device)

for param in source_model.parameters():
    param.requires_grad = False
source_model.eval()

# set up optimizer
# different learning rate for different part of GNN
model_param_group = []
model_param_group.append({"params": finetuned_model.gnn.parameters()})
model_param_group.append({"params": finetuned_model.fc2.parameters(), "lr": finetuned_model.learning_rate})

optimizer = optim.Adam(model_param_group, lr=finetuned_model.learning_rate)
# create intermediate layer getter
return_layers = ['gnn.gnns.4.mlp.2']
return_layers = ['gnn']
# get the output feature map of the mediate layer in full model
source_getter = IntermediateLayerGetter(source_model, return_layers=return_layers)
target_getter = IntermediateLayerGetter(finetuned_model, return_layers=return_layers)

# get regularization for finetune
weights_regularization = FrobeniusRegularization(source_model.gnn, finetuned_model.gnn)
backbone_regularization = lambda x: x
bss_regularization = lambda x: x


''' the proposed method GTOT-tuning'''
backbone_regularization = GTOTRegularization(order=order)

head_regularization = L2Regularization(nn.ModuleList([finetuned_model.gnn]))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=6,
                                                        verbose=False,
                                                        threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                        min_lr=1e-8,
                                                        eps=1e-08)
save_model_name = 'finetuned_model'
stopper = EarlyStopping(mode='lower', patience=patience, filename=save_model_name)


fname = 'finetuning_runs/finetune'
print('tensorboard file', fname)
writer = SummaryWriter(fname)
training_time = Runtime()
test_time = Runtime()
for epoch in range(1, epochs):
    print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]['lr'])
    training_time.epoch_start()
    train_acc, train_loss = train_epoch(finetuned_model, device, train_loader, optimizer,
                                                            weights_regularization,
                                                            backbone_regularization,
                                                            head_regularization, target_getter,
                                                            source_getter, bss_regularization,
                                                            trade_off_backbone, trade_off_head,
                                                            scheduler,
                                                            epoch)
    training_time.epoch_end()

    print("====Evaluation")
    exit()
    val_acc, val_loss = eval(args, model, device, val_loader)
    test_time.epoch_start()
    test_acc, test_loss = eval(args, model, device, test_loader)
    test_time.epoch_end()
    try:
        scheduler.step(-val_acc)
    except:
        scheduler.step()


    writer.add_scalar('data/train auc', train_acc, epoch)
    writer.add_scalar('data/val auc', val_acc, epoch)
    writer.add_scalar('data/test auc', test_acc, epoch)
    writer.add_scalar('data/train loss', train_loss, epoch)
    writer.add_scalar('data/val loss', val_loss, epoch)
    writer.add_scalar('data/test loss', test_loss, epoch)

    if stopper.step(val_acc, finetuned_model, test_score=test_acc, IsMaster=True):
        stopper.report_final_results(i_epoch=epoch)
        break
    stopper.print_best_results(i_epoch=epoch, val_cls_loss=val_loss, train_acc=train_acc, val_score=val_acc,
                                test_socre=test_acc, gnn_type=gnn_config['layer_type'],
                                dataset=dataset, tag=tag)

training_time.print_mean_sum_time(prefix='Training')
test_time.print_mean_sum_time(prefix='Test')


print('tensorboard file is saved in', fname)
writer.close()

print(stopper.best_test_score, stopper.best_epoch, training_time)
test_results = trainer.test(finetuned_model, test_data_loader)




