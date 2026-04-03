import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
from torch_geometric.nn import Sequential
from sklearn.utils.class_weight import compute_class_weight
import lightning.pytorch as pl
import os
import json
import networkx as nx
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import balanced_accuracy_score
import matplotlib as mpl
from statistics import mean, stdev
from torch_geometric.explain import AttentionExplainer, Explainer, ModelConfig
import types
from argparse import ArgumentParser

torch_geometric.seed_everything(42)


def compute_attention_explanations(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_att_base = args.save_dir_att
    mc_dropout = args.mc_dropout
    ood_data = args.ood_data
    
    # Determine final save directory
    save_dir_att = save_dir_att_base

    if os.path.exists(save_dir_att):
        print("Save directory already exists")
        print(save_dir_att)
        return
    os.makedirs(save_dir_att)
    
    fold_list = os.listdir(checkpoint_dir)
    checkpoint_fold_list = [
        os.path.join(checkpoint_dir, fold) for fold in fold_list
    ]
    fold_list.sort()
    checkpoint_fold_list.sort()
    
    att_explainer = AttentionExplainer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine targets (OOD vs ID)
    if ood_data:
        target_names = sorted([p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))])
    else:
        target_names = None
    
    for i, fold in enumerate(fold_list):
        print(f"\n{'='*60}")
        print(f"Fold {i} | MC Dropout: {mc_dropout} | OOD: {ood_data}")
        print(f"{'='*60}")
        
        checkpoint_path = os.path.join(
            checkpoint_fold_list[i], os.listdir(checkpoint_fold_list[i])[0]
        )
        
        # Model hyperparameters
        n_gat_layers = 1
        hidden_dim = 32
        dropout = 0.0
        slope = 0.0025
        pooling_method = "mean"
        norm_method = "batch"
        activation = "leaky_relu"
        n_heads = 9
        lr = 0.0012
        weight_decay = 0.0078
        
        # Determine test targets for this fold
        if ood_data:
            current_targets = target_names
            current_dirs = [os.path.join(data_dir, t) for t in current_targets]
            log_names = [f"{fold}_{t}" for t in current_targets]
        else:
            current_targets = ["id_test"]
            current_dirs = [os.path.join(data_dir, fold)]
            log_names = [fold]
        
        # Load model once per fold
        features_shape = GraphDataset(current_dirs[0])[0].x.shape[-1]
        n_classes = 3

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
        model = model.to(device)
        model.eval()
        
        # Process each target (for OOD: multiple targets; for ID: single target)
        for t_idx, (t_name, t_dir, log_name) in enumerate(zip(current_targets, current_dirs, log_names)):
            print(f"\nTarget: {t_name}")
            
            dataset = GraphDataset(t_dir)
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=2 if not mc_dropout else 0,
                prefetch_factor=20 if not mc_dropout else 0,
            )
            
            config = ModelConfig(
                "multiclass_classification", task_level="graph", return_type="raw"
            )
            explainer = Explainer(
                model,
                algorithm=att_explainer,
                explanation_type="model",
                model_config=config,
                edge_mask_type="object",
            )
            
            # ================================================================
            # BASELINE MODE: Global Averaging
            # ================================================================
            if not mc_dropout:
                edge_connection_dict_all = {}
                edge_connection_dict_preictal = {}
                edge_connection_dict_interictal = {}
                edge_connection_dict_ictal = {}
                interictal_cntr = 0
                preictal_cntr = 0
                ictal_cntr = 0
                
                for n, batch in enumerate(loader):
                    batch = batch.to(device)
                    explanation = explainer(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        target=batch.y,
                        pyg_batch=batch.batch,
                    )
                    
                    for edge_idx in range(explanation.edge_index.size(1)):
                        edge = explanation.edge_index[:, edge_idx].tolist()
                        edge.sort()
                        edge = str(tuple(edge))
                        edge_mask = explanation.edge_mask[edge_idx].item()
                        prediction = torch.argmax(explanation.prediction)
                        
                        if edge in edge_connection_dict_all.keys():
                            edge_connection_dict_all[edge] += edge_mask
                        else:
                            edge_connection_dict_all[edge] = edge_mask
                        
                        if batch.y == 0 and prediction == 0:
                            if edge in edge_connection_dict_preictal.keys():
                                edge_connection_dict_preictal[edge] += edge_mask
                            else:
                                edge_connection_dict_preictal[edge] = edge_mask
                        elif batch.y == 1 and prediction == 1:
                            if edge in edge_connection_dict_ictal.keys():
                                edge_connection_dict_ictal[edge] += edge_mask
                            else:
                                edge_connection_dict_ictal[edge] = edge_mask
                        elif batch.y == 2 and prediction == 2:
                            if edge in edge_connection_dict_interictal.keys():
                                edge_connection_dict_interictal[edge] += edge_mask
                            else:
                                edge_connection_dict_interictal[edge] = edge_mask
                    
                    if batch.y == 0 and prediction == 0:
                        preictal_cntr += 1
                    elif batch.y == 1 and prediction == 1:
                        ictal_cntr += 1
                    elif batch.y == 2 and prediction == 2:
                        interictal_cntr += 1
                    
                    if n % 100 == 0:
                        print(f"  Batch {n}/{len(loader)} done")
                
                # Average across dataset
                edge_connection_dict_all = {
                    key: value / (n + 1)
                    for key, value in edge_connection_dict_all.items()
                }
                edge_connection_dict_interictal = {
                    key: value / max(interictal_cntr, 1)
                    for key, value in edge_connection_dict_interictal.items()
                }
                edge_connection_dict_ictal = {
                    key: value / max(ictal_cntr, 1)
                    for key, value in edge_connection_dict_ictal.items()
                }
                edge_connection_dict_preictal = {
                    key: value / max(preictal_cntr, 1)
                    for key, value in edge_connection_dict_preictal.items()
                }
                
                # Save baseline results
                save_path_fold = os.path.join(save_dir_att, f"fold_{i}")
                if not os.path.exists(save_path_fold):
                    os.makedirs(save_path_fold)
                
                file_prefix = f"{t_name}_" if ood_data else ""
                
                with open(
                    os.path.join(save_path_fold, f"{file_prefix}edge_connection_dict_all.json"), "w"
                ) as f:
                    json.dump(edge_connection_dict_all, f)
                with open(
                    os.path.join(save_path_fold, f"{file_prefix}edge_connection_dict_interictal.json"), "w"
                ) as f:
                    json.dump(edge_connection_dict_interictal, f)
                with open(
                    os.path.join(save_path_fold, f"{file_prefix}edge_connection_dict_ictal.json"), "w"
                ) as f:
                    json.dump(edge_connection_dict_ictal, f)
                with open(
                    os.path.join(save_path_fold, f"{file_prefix}edge_connection_dict_preictal.json"), "w"
                ) as f:
                    json.dump(edge_connection_dict_preictal, f)
                
                print(f"  Baseline processing complete for {t_name}")
            
            # ================================================================
            # MC DROPOUT MODE: Instance-Level Stochastic Explanations
            # ================================================================
            else:
                # Enable MC Dropout
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout') or 'GAT' in m.__class__.__name__:
                        m.train()
                        m.eval = types.MethodType(lambda self: self.train(), m)
                
                save_path_fold = os.path.join(save_dir_att, f"fold_{i}")
                if not os.path.exists(save_path_fold):
                    os.makedirs(save_path_fold)
                
                sample_counter = 0
                
                for batch_idx, batch in enumerate(loader):
                    batch = batch.to(device)
                    true_label = batch.y.item()
                    
                    # Collect explanations from T=50 MC passes
                    all_edge_masks = []
                    pred_labels = []
                    
                    for t in range(50):
                        explanation = explainer(
                            x=batch.x,
                            edge_index=batch.edge_index,
                            target=batch.y,
                            pyg_batch=batch.batch,
                        )
                        
                        edge_masks = explanation.edge_mask  # Shape: [E]
                        all_edge_masks.append(edge_masks.cpu())
                        
                        pred_label = torch.argmax(explanation.prediction).item()
                        pred_labels.append(pred_label)
                    
                    # Stack: [T, E]
                    all_edge_masks = torch.stack(all_edge_masks, dim=0)  # [T, E]
                    
                    # Majority vote for prediction
                    pred_label_mode = torch.mode(torch.tensor(pred_labels))[0].item()
                    
                    # Compute mean and std
                    edge_mask_mean = all_edge_masks.mean(dim=0)  # [E]
                    edge_mask_std = all_edge_masks.std(dim=0)    # [E]
                    
                    
                    # Create sample ID matching compute_uncertainty_metrics.py
                    sample_id = f"{fold}_{t_name}_{batch_idx}"
                    
                    # Save instance-level data
                    sample_data = {
                        "sample_id": sample_id,
                        "true_label": true_label,
                        "pred_label_mode": pred_label_mode,
                        "edge_mask_mean": edge_mask_mean.numpy(),
                        "edge_mask_std": edge_mask_std.numpy(),
                        "edge_mask_all": all_edge_masks.numpy(),
                    }
                    
                    file_prefix = f"{t_name}_" if ood_data else ""
                    save_path = os.path.join(
                        save_path_fold,
                        f"sample_{file_prefix}{batch_idx}.pt"
                    )
                    torch.save(sample_data, save_path)
                    
                    if batch_idx % 100 == 0:
                        print(f"  MC Sample {batch_idx}/{len(loader)} done")
                    
                    sample_counter += 1
                
                print(f"  MC processing complete: {sample_counter} samples for {t_name}")
        
        print(f"Fold {i} complete\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing data"
    )
    parser.add_argument(
        "--save_dir_att",
        type=str,
        required=True,
        help="Base directory to save attention explanations"
    )
    parser.add_argument(
        "--mc_dropout",
        action="store_true",
        default=False,
        help="Enable MC Dropout for stochastic explanations"
    )
    parser.add_argument(
        "--ood_data",
        action="store_true",
        default=False,
        help="Use OOD data targets instead of ID test data"
    )
    args = parser.parse_args()
    compute_attention_explanations(args)