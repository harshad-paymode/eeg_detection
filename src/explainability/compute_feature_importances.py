import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import logging
import json
from torch_geometric import seed_everything
from argparse import ArgumentParser
from torch_geometric.explain import GNNExplainer, Explainer, ModelConfig
import types
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

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
    
    os.makedirs(save_dir_importances, exist_ok=True)
    
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
        print(f"Fold {fold} (Index {i}) | MC Dropout: {mc_dropout} | OOD: {ood_data}")
       
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
            dropout_on=False,
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
            # collect all samples into this list for mc dropout
            all_samples = []
            print(f"\nTarget: {t_name}")
            
            dataset = GraphDataset(t_dir)
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                prefetch_factor=None,
            )
            
            gnn_explainer = GNNExplainer(epochs=200, lr=0.01)
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
                
                batch_count = 0
                for n, batch in enumerate(loader):
                    batch_unpacked = batch.to(device)
                    explanation = explainer(
                        x=batch_unpacked.x,
                        edge_index=batch_unpacked.edge_index,
                        target=batch_unpacked.y,
                        pyg_batch=batch_unpacked.batch,
                    )
                    prediction = torch.argmax(explanation.prediction)
                    
                    # BUG FIX: Added .detach() to prevent memory leak
                    sum_masks += explanation.node_mask.detach()
                    
                    # BUG FIX: Added .item() to safely check condition
                    if batch_unpacked.y.item() == 0 and prediction.item() == 0:
                        preictal_masks += explanation.node_mask.detach()
                        preictal_cntr += 1
                    elif batch_unpacked.y.item() == 1 and prediction.item() == 1:
                        ictal_masks += explanation.node_mask.detach()
                        ictal_cntr += 1
                    elif batch_unpacked.y.item() == 2 and prediction.item() == 2:
                        interictal_masks += explanation.node_mask.detach()
                        interictal_cntr += 1
                    
                    batch_count += 1
                    if batch_count % 100 == 0:
                        print(f"  Batch {batch_count}/{len(loader)} done")
                
                # Average across dataset
                sum_masks /= batch_count
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
              
                os.makedirs(save_path_fold, exist_ok=True)
                
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
                
                print(f"  Baseline processing complete for {t_name} ({batch_count} batches)")
            
            # ================================================================
            # MC DROPOUT MODE: Save mask for every sample separately
            # 1. Here only the masks are calculated for MC dropout
            # 2. The predictions will be cross referenced from compute_uncertainty_matrix results using sample id
            # ================================================================
            else:
                save_path_fold = os.path.join(save_dir_importances, f"fold_{i}")
                os.makedirs(save_path_fold, exist_ok=True)
                
                sample_counter = 0
                
                for batch_idx, batch in enumerate(loader):
                    batch_unpacked = batch.to(device)
                    
                    # 1. Run Explainer EXACTLY ONCE
                    explanation = explainer(
                        x=batch_unpacked.x, 
                        edge_index=batch_unpacked.edge_index, 
                        target=batch_unpacked.y, 
                        pyg_batch=batch_unpacked.batch
                    )

                    # Pure PyTorch tensor, no .numpy() conversions
                    node_mask_base = explanation.node_mask.detach().cpu()
                    
                    # 4. Save dictionary
                    sample_data = {
                        "sample_id": f"{fold}_{t_name}_{batch_idx}",
                        "true_label": batch_unpacked.y.item(),
                        "pred_label": torch.argmax(explanation.prediction).item(),
                        "node_mask_base": node_mask_base,
                    }
                    
                    all_samples.append(sample_data)
                    
                    sample_counter += 1
                    if sample_counter % 100 == 0:
                        print(f"MC Sample {sample_counter}/{len(loader)} done")
                
                print(f"MC processing complete: {sample_counter} samples for {t_name}")

                file_prefix = f"{t_name}" if ood_data else "test_data"
                torch.save(all_samples, os.path.join(save_path_fold, f"{file_prefix}.pt"))
            
        print(f"Fold {fold} complete\n")


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