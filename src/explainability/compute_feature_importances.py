import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import json
from torch_geometric import seed_everything
from argparse import ArgumentParser
from torch_geometric.explain import GNNExplainer, Explainer, ModelConfig
import types

seed_everything(42)



def compute_feature_importances(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_importances_base = args.save_dir_importances
    mc_dropout = args.mc_dropout
    ood_data = args.ood_data
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine final save directory
    save_dir_importances = save_dir_importances_base
    
    os.makedirs(save_dir_importances,exist_ok=True)
    
    fold_list = os.listdir(checkpoint_dir)
    checkpoint_fold_list = [
        os.path.join(checkpoint_dir, fold) for fold in fold_list
    ]
    fold_list.sort()
    checkpoint_fold_list.sort()
    
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
                prefetch_factor=20 if not mc_dropout else None,
            )
            
            gnn_explainer = GNNExplainer(epochs=50, lr=0.01)
            config = ModelConfig(
                "multiclass_classification", task_level="graph", return_type="raw"
            )
            explainer = Explainer(
                model,
                algorithm=gnn_explainer,
                explanation_type="model",
                model_config=config,
                node_mask_type="attributes",
                edge_mask_type="object",
            )
            
            # ================================================================
            # BASELINE MODE: Global Averaging
            # ================================================================
            if not mc_dropout:
                sum_masks = torch.zeros((18, 10)).to(device)
                interictal_masks = torch.zeros((18, 10)).to(device)
                ictal_masks = torch.zeros((18, 10)).to(device)
                preictal_masks = torch.zeros((18, 10)).to(device)
                interictal_cntr = 0
                preictal_cntr = 0
                ictal_cntr = 0
                
                for n, batch in enumerate(loader):
                    batch_unpacked = batch.to(device)
                    explanation = explainer(
                        x=batch_unpacked.x,
                        edge_index=batch_unpacked.edge_index,
                        target=batch_unpacked.y,
                        pyg_batch=batch_unpacked.batch,
                    )
                    prediction = torch.argmax(explanation.prediction)
                    
                    sum_masks += explanation.node_mask
                    
                    if batch_unpacked.y == 0 and prediction == 0:
                        preictal_masks += explanation.node_mask
                        preictal_cntr += 1
                    elif batch_unpacked.y == 1 and prediction == 1:
                        ictal_masks += explanation.node_mask
                        ictal_cntr += 1
                    elif batch_unpacked.y == 2 and prediction == 2:
                        interictal_masks += explanation.node_mask
                        interictal_cntr += 1
                    
                    if n % 100 == 0 and n != 0:
                        print(f"  Batch {n}/{len(loader)} done")
                
                # Average across dataset
                sum_masks /= (n + 1)
                interictal_masks /= max(interictal_cntr, 1)
                ictal_masks /= max(ictal_cntr, 1)
                preictal_masks /= max(preictal_cntr, 1)
                
                # Create final explanation objects
                final_explanation_sum = explanation.clone()
                final_explanation_interictal = explanation.clone()
                final_explanation_preictal = explanation.clone()
                final_explanation_ictal = explanation.clone()
                
                final_explanation_sum.node_mask = sum_masks
                final_explanation_interictal.node_mask = interictal_masks
                final_explanation_preictal.node_mask = preictal_masks
                final_explanation_ictal.node_mask = ictal_masks
                
                # Save baseline results
                save_path_fold = os.path.join(save_dir_importances, f"fold_{i}")
                if not os.path.exists(save_path_fold):
                    os.makedirs(save_path_fold,exist_ok=True)
                
                file_prefix = f"{t_name}_" if ood_data else ""
                
                torch.save(
                    final_explanation_sum,
                    os.path.join(save_path_fold, f"{file_prefix}final_explanation_sum.pt"),
                )
                torch.save(
                    final_explanation_interictal,
                    os.path.join(save_path_fold, f"{file_prefix}final_explanation_interictal.pt"),
                )
                torch.save(
                    final_explanation_preictal,
                    os.path.join(save_path_fold, f"{file_prefix}final_explanation_preictal.pt"),
                )
                torch.save(
                    final_explanation_ictal,
                    os.path.join(save_path_fold, f"{file_prefix}final_explanation_ictal.pt"),
                )
                
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
                
                save_path_fold = os.path.join(save_dir_importances, f"fold_{i}")
                if not os.path.exists(save_path_fold):
                    os.makedirs(save_path_fold,exist_ok=True)
                
                sample_counter = 0
                
                for batch_idx, batch in enumerate(loader):
                    batch = batch.to(device)
                    true_label = batch.y.item()
                    num_nodes = batch.x.size(0)
                    
                    # Collect explanations from T=50 MC passes
                    all_node_masks = []
                    pred_labels = []
                    
                    for t in range(50):
                        explanation = explainer(
                            x=batch.x,
                            edge_index=batch.edge_index,
                            target=batch.y,
                            pyg_batch=batch.batch,
                        )
                        
                        node_masks = explanation.node_mask  # Shape: [N, F]
                        
                        # Aggregate node importance: mean across features -> [N]
                        node_importance = node_masks.mean(dim=1) if node_masks.dim() > 1 else node_masks
                        all_node_masks.append(node_importance)
                        
                        pred_label = torch.argmax(explanation.prediction).item()
                        pred_labels.append(pred_label)
                    
                    # Stack: [T, N] and [T, E]
                    all_node_masks = torch.stack(all_node_masks, dim=0).cpu()  # [T, N]
                    
                    # Majority vote for prediction
                    pred_label_mode = torch.mode(torch.tensor(pred_labels))[0].item()
                    
                    # Compute mean and std
                    node_mask_mean = all_node_masks.mean(dim=0)  # [N]
                    node_mask_std = all_node_masks.std(dim=0)    # [N]
                    
                    # Create sample ID matching compute_uncertainty_metrics.py
                    sample_id = f"{fold}_{t_name}_{batch_idx}"
                    
                    # Save instance-level data
                    sample_data = {
                        "sample_id": sample_id,
                        "true_label": true_label,
                        "pred_label_mode": pred_label_mode,
                        "node_mask_mean": node_mask_mean.numpy(),
                        "node_mask_std": node_mask_std.numpy(),
                        "node_mask_all": all_node_masks.numpy(),
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
        "--save_dir_importances",
        type=str,
        required=True,
        help="Base directory to save feature importances"
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
    compute_feature_importances(args)