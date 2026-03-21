import torch
from torch_geometric.loader import DataLoader
import numpy as np
from models import GATv2Lightning
from utils.dataloader_utils import GraphDataset
import os
import json
from argparse import ArgumentParser
from statistics import mean, stdev
from torchmetrics.classification import MulticlassCalibrationError, MulticlassBrierScore
from sklearn.metrics import auc


def enable_dropout(model):
    """Force dropout layers to stay active during evaluation."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def compute_aurc(uncertainties, errors):
    """Calculates Area Under the Risk-Coverage Curve."""
    sort_idx = np.argsort(uncertainties)
    errors_sorted = errors[sort_idx]
    
    coverages = np.arange(1, len(errors) + 1) / len(errors)
    risks = np.cumsum(errors_sorted) / np.arange(1, len(errors) + 1)
    
    return auc(coverages, risks)


def compute_uncertainty_metrics(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_metrics = args.save_dir_metrics
    mc_passes = args.mc_passes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(save_dir_metrics):
        print("Save directory already exists")
        print(save_dir_metrics)
        return
    os.makedirs(save_dir_metrics)
    
    fold_list = os.listdir(checkpoint_dir)
    checkpoint_fold_list = [os.path.join(checkpoint_dir, fold) for fold in fold_list]
    data_fold_list = [os.path.join(data_dir, fold) for fold in fold_list]
    fold_list.sort()
    data_fold_list.sort()
    checkpoint_fold_list.sort()

    ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='l1').to(device)
    brier_metric = MulticlassBrierScore(num_classes=3).to(device)

    summary_ece = []
    summary_brier = []
    summary_aurc = []

    for n, fold in enumerate(fold_list):
        checkpoint_path = os.path.join(
            checkpoint_fold_list[n], os.listdir(checkpoint_fold_list[n])[0]
        )

        n_gat_layers = 1
        hidden_dim = 32
        dropout = 0.4  # Important: Must be > 0 for MC Dropout to work
        slope = 0.0025
        pooling_method = "mean"
        norm_method = "batch"
        activation = "leaky_relu"
        n_heads = 9
        lr = 0.0012
        weight_decay = 0.0078
        dataset = GraphDataset(data_fold_list[n])
        n_classes = 3
        features_shape = dataset[0].x.shape[-1]

        model = GATv2Lightning.load_from_checkpoint(
            checkpoint_path,
            in_features=features_shape,
            n_classes=n_classes,
            n_gat_layers=n_gat_layers,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            slope=slope,
            dropout=dropout,
            pooling_method=pooling_method,
            activation=activation,
            norm_method=norm_method,
            lr=lr,
            weight_decay=weight_decay,
            map_location=device,
        )
        
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        model.eval()
        enable_dropout(model)

        all_mean_probs = []
        all_uncertainties = []
        all_targets = []
        all_errors = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                
                mc_preds = []
                for _ in range(mc_passes):
                    logits = model.forward(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    mc_preds.append(probs.unsqueeze(0))
                
                mc_preds = torch.cat(mc_preds, dim=0)
                mean_probs = mc_preds.mean(dim=0)
                
                # Predictive Entropy (Uncertainty)
                uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
                
                predictions = mean_probs.argmax(dim=1)
                errors = (predictions != batch.y).float()
                
                all_mean_probs.append(mean_probs)
                all_uncertainties.append(uncertainty)
                all_targets.append(batch.y)
                all_errors.append(errors)

        fold_probs = torch.cat(all_mean_probs)
        fold_targets = torch.cat(all_targets)
        fold_uncertainties = torch.cat(all_uncertainties).cpu().numpy()
        fold_errors = torch.cat(all_errors).cpu().numpy()

        fold_ece = ece_metric(fold_probs, fold_targets).item()
        fold_brier = brier_metric(fold_probs, fold_targets).item()
        fold_aurc = compute_aurc(fold_uncertainties, fold_errors)

        summary_ece.append(fold_ece)
        summary_brier.append(fold_brier)
        summary_aurc.append(fold_aurc)

        # saving fold results
        fold_results = {
            "ECE": fold_ece,
            "Brier_Score": fold_brier,
            "AURC": fold_aurc,
        }
        
        with open(
            os.path.join(save_dir_metrics, f"fold_{n}_uncertainty_results.json"), "w"
        ) as f:
            json.dump(fold_results, f)

    # saving summary results
    summary_results = {
        "ECE": mean(summary_ece),
        "ECE_std": stdev(summary_ece),
        "Brier_Score": mean(summary_brier),
        "Brier_Score_std": stdev(summary_brier),
        "AURC": mean(summary_aurc),
        "AURC_std": stdev(summary_aurc),
    }

    with open(
        os.path.join(save_dir_metrics, f"summary_uncertainty_results.json"), "w"
    ) as f:
        json.dump(summary_results, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir_metrics", type=str, required=True)
    parser.add_argument("--mc_passes", type=int, default=30)
    args = parser.parse_args()
    compute_uncertainty_metrics(args)