# THIS WORK IS FROM JOHN BRADSHAW
import time
import typing
import collections
import itertools
from mol2graph import Graphs
from dataclasses import dataclass

import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.nn import functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

from ignite.contrib.handlers import ProgressBar


class GNN(nn.Module):
    def __init__(self, node_feature_dimension, num_propagation_steps :int =4):
        super().__init__()

        self.num_propagation_steps = num_propagation_steps
        # called T above.

        # Our sub modules:
        self.message_projection = nn.Linear(node_feature_dimension, node_feature_dimension, bias=False)
        self.update_gru = nn.GRUCell(input_size=node_feature_dimension,
                                     hidden_size=node_feature_dimension, bias=True)
        self.attn_net = nn.Linear(node_feature_dimension, 1)
        self.proj_net = nn.Linear(node_feature_dimension, node_feature_dimension)
        self.final_lin = nn.Linear(node_feature_dimension, 1)

    def forward(self, graphs_in: Graphs):
        """
        Produces a column vector of predictions, with each element in this vector a prediction
        for each marked graph in `graphs_in`.

        In the comments below N is the number of nodes in graph_in (across all graphs),
        d the feature dimension, and G is the number of individual molecular graphs.
        """
        # 1. Message passing and updating
        m = graphs_in.node_features  # shape: [N, d]

        for t in range(self.num_propagation_steps):
            projs = self.message_projection(m)  # [N, d]

            # Update the node embeddings (eqn 1 above)
            # 1a. compute the sum for each node
            msgs = torch.zeros_like(m)  # [N, d]
            msgs.index_add_(0, graphs_in.edge_list[:, 0], projs.index_select(0, graphs_in.edge_list[:, 1]))

            # 1b. update the embeddings via GRU cell
            m = self.update_gru(msgs, m)  # [N, d]

        # 2. Aggregation (eqn 2 above)
        # a compute weighted embeddings
        attn_coeffs = torch.sigmoid(self.attn_net(m))  # [N, 1]
        proj_embeddings = self.proj_net(m)  # [N, d']
        weighted_embeddings = attn_coeffs * proj_embeddings

        # perform the sum
        graph_embedding = torch.zeros(graphs_in.num_graphs, weighted_embeddings.shape[1],
                                      device=m.device, dtype=m.dtype)
        graph_embedding.index_add_(0, graphs_in.node_to_graph_id, weighted_embeddings)  # [G, d']

        # 3. Final linear projection.
        final_prediction = self.final_lin(graph_embedding)  # [G, 1]
        return final_prediction


@dataclass
class TrainParams:
    batch_size: int = 64
    val_batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: typing.Optional[str] = 'cpu'


def train_neural_network(train_dataset: np.ndarray, val_dataset: np.ndarray,
                          smiles_col:str, regression_column:str,
                         transform: typing.Callable,
                         neural_network: nn.Module,
                         params: typing.Optional[TrainParams]=None,
                         collate_func: typing.Optional[typing.Callable]=None):
    """
    Trains a PyTorch NN module on train dataset, validates it each epoch and returns a series of useful metrics
    for further analysis. Note the networks parameters will be changed in place.
    :param train_df: data to use for training.
    :param val_df: data to use for validation.
    :param smiles_col: column name for SMILES data in Dataframe
    :param regression_column: column name for the data we want to regress to.
    :param transform: the transform to apply to the datasets to create new ones suitable for working with neural network
    :param neural_network: the PyTorch nn.Module to train
    :param params: the training params eg number of epochs etc.
    :param collate_func: collate_fn to pass to dataloader constructor. Leave as None to use default.
    """
    if params is None:
        params = TrainParams()



    # Update the train and valid datasets with new parameters
    train_dataset = SmilesRegressionDataset.create_from_df(train_df, smiles_col, regression_column, transform=transform)
    val_dataset = SmilesRegressionDataset.create_from_df(val_df, smiles_col, regression_column, transform=transform)
    print(f"Train dataset is of size {len(train_dataset)} and valid of size {len(val_dataset)}")

    # Put into dataloaders
    train_dataloader = data.DataLoader(train_dataset, params.batch_size, shuffle=True,
                                       collate_fn=collate_func, num_workers=1)
    val_dataloader = data.DataLoader(val_dataset, params.val_batch_size, shuffle=False, collate_fn=collate_func,
                                       num_workers=1)

    # Optimizer
    optimizer = optim.Adam(neural_network.parameters(), lr=params.learning_rate)

    # Work out what device we're going to run on (ie CPU or GPU)
    device = params.device

    # We're going to use PyTorch Ignite to take care of the majority of the training boilerplate for us
    # see https://pytorch.org/ignite/
    # in particular we are going to follow the example
    # https://github.com/pytorch/ignite/blob/53190db227f6dda8980d77fa5351fa3ddcdec6fb/examples/contrib/mnist/mnist_with_tqdm_logger.py
    def prepare_batch(batch, device, non_blocking):
        x, y = batch
        return x.to(device), y.to(device)

    trainer = create_supervised_trainer(neural_network, optimizer, F.mse_loss, device=device, prepare_batch=prepare_batch)
    evaluator = create_supervised_evaluator(neural_network,
                                            metrics={'loss': Loss(F.mse_loss)},
                                            device=device, prepare_batch=prepare_batch)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names='all')

    train_loss_list = []
    val_lost_list = []
    val_times_list = []

    @trainer.on(Events.EPOCH_COMPLETED | Events.STARTED)
    def log_training_results(engine):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        pbar.log_message("Epoch - {}".format(engine.state.epoch))
        pbar.log_message(
            "Training Results - Epoch: {}  Avg loss: {:.2f}"
                .format(engine.state.epoch, loss)
        )
        train_loss_list.append(loss)

    @trainer.on(Events.EPOCH_COMPLETED | Events.STARTED)
    def log_validation_results(engine):
        s_time = time.time()
        evaluator.run(val_dataloader)
        e_time = time.time()
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        pbar.log_message(
            "Validation Results - Epoch: {} Avg loss: {:.2f}"
                .format(engine.state.epoch, loss))

        pbar.n = pbar.last_print_n = 0
        val_lost_list.append(loss)
        val_times_list.append(e_time - s_time)

    # We can now train our network!
    trainer.run(train_dataloader, max_epochs=params.num_epochs)

    # Having trained it wee are now also going to run through the validation set one
    # last time to get the actual predictions
    val_predictions = []
    neural_network.eval()
    for batch in val_dataloader:
        x, y = batch
        x = x.to(device)
        y_pred = neural_network(x)
        assert (y_pred.shape) == (y.shape), "If this is not true then would cause problems in loss"
        val_predictions.append(y_pred.cpu().detach().numpy())
    neural_network.train()
    val_predictions = np.concatenate(val_predictions)

    # Create a table of useful metrics (as part of the information we return)
    total_number_params = sum([v.numel() for v in  neural_network.parameters()])
    out_table = [
        ["Num params", f"{total_number_params:.2e}"],
        ["Minimum train loss", f"{np.min(train_loss_list):.3f}"],
        ["Mean validation time", f"{np.mean(val_times_list):.3f}"],
        ["Minimum validation loss", f"{np.min(val_lost_list):.3f}"]
     ]

    # We will create a dictionary of results.
    results = dict(
        train_loss_list=train_loss_list,
        val_lost_list=val_lost_list,
        val_times_list=val_times_list,
        out_table=out_table,
        val_predictions=val_predictions
    )
    return results


def plot_train_and_val_using_altair(train_loss, val_loss):
    """
    Plots the train and validation loss using Altair -- see https://altair-viz.github.io/gallery/multiline_tooltip.html
    This should result in an interactive plot which we can mouseover.
    """
    assert len(train_loss) == len(val_loss)
    source = pd.DataFrame.from_dict({"train_loss": train_loss, "val_loss": val_loss, 'epoch': np.arange(len(train_loss))})
    source = source.melt('epoch', var_name='category', value_name='loss')

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['epoch'], empty='none')

    # The basic line
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x='epoch:Q',
        y='loss:Q',
        color='category:N'
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x='epoch:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'loss:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='epoch:Q',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    return alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=600, height=300
    )