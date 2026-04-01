import torch
import lightning.pytorch as pl
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.nn import GATv2Conv, Linear, Sequential
from typing import List, Union, Tuple, Callable
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torch_geometric.nn import TransformerConv
from torchmetrics import Specificity, Recall, AUROC, F1Score


class GATv2Lightning(pl.LightningModule):
    """Lightning Module implementing GATv2 network used for experiments."""

    def __init__(
        self,
        in_features: int,
        n_gat_layers: int = 2,
        hidden_dim: int = 32,
        n_heads: int = 4,
        dropout_on: bool = False,
        slope: float = 0.01,
        pooling_method: str = "mean",
        activation: str = "leaky_relu",
        norm_method: str = "batch",
        n_classes: int = 2,
        fft_mode: bool = False,
        lr=0.00001,
        weight_decay=0.0001,
        class_weights=None,
    ):
        super(GATv2Lightning, self).__init__()
        assert n_classes > 1, "n_classes must be greater than 1"
        self.classification_mode = "multiclass" if n_classes > 2 else "binary"
        assert activation in [
            "leaky_relu",
            "relu",
        ], 'Activation must be either "leaky_relu" or "relu"'
        assert norm_method in [
            "batch",
            "layer",
        ], 'Norm_method must be either "batch" or "layer"'
        assert pooling_method in [
            "mean",
            "max",
            "add",
        ], "Pooling_method must be either 'mean', 'max', or 'add'"
        if class_weights is not None:
            if n_classes > 2:
                assert (
                    len(class_weights) == n_classes
                ), "Number of class weights must match number of classes"
            else:
                assert (
                    len(class_weights) == 1
                ), "Only one class weight must be provided for binary classification"
        act_fn = (
            nn.LeakyReLU(slope, inplace=True)
            if activation == "leaky_relu"
            else nn.ReLU(inplace=True)
        )
        norm_layer = (
            BatchNorm(hidden_dim * n_heads)
            if norm_method == "batch"
            else LayerNorm(hidden_dim * n_heads)
        )
    
        classifier_out_neurons = n_classes if n_classes > 2 else 1
        feature_extractor_list: List[
            Union[Tuple[Callable, str], Callable]
        ] = []
        for i in range(n_gat_layers):
            # feature_extractor_list.append(
            #     (
            #         GATv2Conv(
            #             in_features if i == 0 else hidden_dim * n_heads,
            #             hidden_dim,  # / (2**i) if i != 0 else hidden_dim,
            #             heads=n_heads,
            #             negative_slope=slope,
            #             add_self_loops=True,
            #             dropout = 0.1*dropout_on,
            #             improved = True,
            #             edge_dim=1,
            #         ),
            #         "x, edge_index, edge_attr -> x",
            #     )
            # )
            feature_extractor_list.append(
            (
                TransformerConv(
                    in_channels=in_features if i == 0 else hidden_dim * n_heads,
                    out_channels=hidden_dim,
                    heads=n_heads,
                    dropout=0.1 * dropout_on,
                    edge_dim=1,       # Critical since you have edge attributes
                    beta=True,        # ENABLE THIS: Adds a gating mechanism (improves deep GNNs)
                    root_weight=True  # Replaces the need for add_self_loops
                ),
                "x, edge_index, edge_attr -> x",
            )
        )
            
            feature_extractor_list.append(norm_layer)
            feature_extractor_list.append(act_fn)
        self.feature_extractor = Sequential(
            "x, edge_index, edge_attr", feature_extractor_list
        )

        self.classifier = nn.Sequential(
            Linear(
                hidden_dim * n_heads, 512, weight_initializer="kaiming_uniform"
            ),
            nn.Dropout(0.4*dropout_on),  
            act_fn,
            Linear(512, 256, weight_initializer="kaiming_uniform"),
            nn.Dropout(0.2*dropout_on),
            act_fn,
            Linear(256, 128, weight_initializer="kaiming_uniform"),
            nn.Dropout(0.2*dropout_on),
            act_fn,
            Linear(128, 128, weight_initializer="kaiming_uniform"),
            nn.Dropout(0.2*dropout_on),
            act_fn,
            Linear(
                128,
                classifier_out_neurons,
                weight_initializer="kaiming_uniform",
            ),
        )

        if pooling_method == "mean":
            self.pooling_method = global_mean_pool
        elif pooling_method == "max":
            self.pooling_method = global_max_pool
        elif pooling_method == "add":
            self.pooling_method = global_add_pool
        self.n_classes = n_classes
        self.fft_mode = fft_mode
        self.lr = lr
        self.weight_decay = weight_decay
        if class_weights is None:
            class_weights = torch.ones(n_classes)
        self.class_weights = class_weights
        if self.classification_mode == "multiclass":
            self.f1_score = F1Score(task="multiclass", num_classes=n_classes)
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
            self.recall = Recall(
                task="multiclass", num_classes=n_classes, threshold=0.5
            )
            self.specificity = Specificity(
                task="multiclass", num_classes=n_classes, threshold=0.5
            )
            self.auroc = AUROC(task="multiclass", num_classes=n_classes)
        elif self.classification_mode == "binary":
            self.f1_score = F1Score(task="binary", threshold=0.5)
            self.loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            self.recall = Recall(task="binary", threshold=0.5)
            self.specificity = Specificity(task="binary", threshold=0.5)
            self.auroc = AUROC(task="binary")

        self.temperature = 1.0
        
        self.training_step_outputs: List[torch.Tensor] = []
        self.training_step_gt: List[torch.Tensor] = []
        self.validation_step_outputs: List[torch.Tensor] = []
        self.validation_step_gt: List[torch.Tensor] = []
        self.test_step_outputs: List[torch.Tensor] = []
        self.test_step_gt: List[torch.Tensor] = []

    def forward(self, x, edge_index, pyg_batch, edge_attr=None):
        h = self.feature_extractor(x, edge_index=edge_index, edge_attr=None)
        h = self.pooling_method(h, pyg_batch)
        h = self.classifier(h)
        return h / self.temperature

    def unpack_data_batch(self, data_batch):
        x = data_batch.x
        edge_index = data_batch.edge_index
        y = (
            data_batch.y.long()
            if self.classification_mode == "multiclass"
            else data_batch.y
        )
        pyg_batch = data_batch.batch
        try:
            edge_attr = data_batch.edge_attr.float()
        except AttributeError:
            edge_attr = None

        return x, edge_index, y, pyg_batch, edge_attr

    def training_step(self, batch, batch_idx):
        x, edge_index, y, pyg_batch, edge_attr = self.unpack_data_batch(batch)
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
        loss = self.loss(y_hat, y)

        self.training_step_outputs.append(y_hat)
        self.training_step_gt.append(y)
        batch_size = pyg_batch.max() + 1
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_end(self):
        training_step_outputs = torch.cat(self.training_step_outputs)
        training_step_gt = torch.cat(self.training_step_gt)
        rec = self.recall(training_step_outputs, training_step_gt)
        spec = self.specificity(training_step_outputs, training_step_gt)
        auroc = self.auroc(training_step_outputs, training_step_gt)
        f1_score = self.f1_score(training_step_outputs, training_step_gt)
        self.log_dict(
            {
                "train_sensitivity": rec,
                "train_specificity": spec,
                "train_AUROC": auroc,
                "train_f1_score": f1_score,
            },
            logger=True,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.training_step_outputs.clear()
        self.training_step_gt.clear()

    def validation_step(self, batch, batch_idx):
        x, edge_index, y, pyg_batch, edge_attr = self.unpack_data_batch(batch)
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
        loss = self.loss(y_hat, y)
        self.validation_step_outputs.append(y_hat)
        self.validation_step_gt.append(y)
        batch_size = pyg_batch.max() + 1
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        validation_step_outputs = torch.cat(self.validation_step_outputs)
        validation_step_gt = torch.cat(self.validation_step_gt)
        rec = self.recall(validation_step_outputs, validation_step_gt)
        spec = self.specificity(validation_step_outputs, validation_step_gt)
        auroc = self.auroc(validation_step_outputs, validation_step_gt)
        f1_score = self.f1_score(validation_step_outputs, validation_step_gt)
        self.log_dict(
            {
                "val_sensitivity": rec,
                "val_specificity": spec,
                "val_AUROC": auroc,
                "val_f1_score": f1_score,
            },
            logger=True,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.validation_step_outputs.clear()
        self.validation_step_gt.clear()

    def test_step(self, batch, batch_idx):
        x, edge_index, y, pyg_batch, edge_attr = self.unpack_data_batch(batch)
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
        loss = self.loss(y_hat, y)
        self.test_step_outputs.append(y_hat)
        self.test_step_gt.append(y)
        batch_size = pyg_batch.max() + 1
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def on_test_epoch_end(self) -> None:
        test_step_outputs = torch.cat(self.test_step_outputs)
        test_step_gt = torch.cat(self.test_step_gt)
        rec = self.recall(test_step_outputs, test_step_gt)
        spec = self.specificity(test_step_outputs, test_step_gt)
        auroc = self.auroc(test_step_outputs, test_step_gt)
        f1_score = self.f1_score(test_step_outputs, test_step_gt)
        self.log_dict(
            {
                "test_sensitivity": rec,
                "test_specificity": spec,
                "test_AUROC": auroc,
                "test_f1_score": f1_score,
            },
            logger=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.test_step_outputs.clear()
        self.test_step_gt.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch.x
        edge_index = batch.edge_index
        pyg_batch = batch.batch
        edge_attr = batch.edge_attr
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

