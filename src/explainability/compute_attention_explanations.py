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
from torch_geometric import seed_everything
seed_everything(42)


def compute_attention_explanations(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_att_base = args.save_dir_att
    mc_dropout = args.mc_dropout
    ood_data = args.ood_data
    max_batches = args.max_batches
    
    # Determine final save directory
    save_dir_att = save_dir_att_base

    os.makedirs(save_dir_att, exist_ok=True)
    
    fold_list = os.listdir(checkpoint_dir)
    checkpoint_fold_list = [
        os.path.join(checkpoint_dir, fold) for fold in fold_list
    ]
    fold_list.sort()
    checkpoint_fold_list.sort()
    
    # HARDCODED: For OOD data, only process fold_0 and fold_1
    if ood_data:
        fold_list = [f for f in fold_list if f in ['fold_0', 'fold_1', 'fold_2','fold_3']]
        checkpoint_fold_list = [
            os.path.join(checkpoint_dir, fold) for fold in fold_list
        ]
        print(f"\n[OOD MODE] Processing only folds: {fold_list}")
    
    att_explainer = AttentionExplainer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine targets (OOD vs ID)
    if ood_data:
        target_names = sorted([p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))])
    else:
        target_names = None
    
    for i, fold in enumerate(fold_list):
        print(f"\n{'='*60}")
        print(f"Fold {fold} (Index {i}) | MC Dropout: {mc_dropout} | OOD: {ood_data}")
        if not ood_data:
            print(f"Max Batches: {max_batches}")
        else:
            print(f"Max Batches: ALL (OOD - patient-specific data)")
        print(f"{'='*60}")
        
        checkpoint_path = os.path.join(
            checkpoint_fold_list[i], os.listdir(checkpoint_fold_list[i])[0]
        )
        
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
            n_gat_layers=1,
            hidden_dim=32,
            n_heads=9,
            slope=0.0025,
            dropout=mc_dropout,
            pooling_method="mean",
            activation="leaky_relu",
            norm_method="batch",
            lr=0.0012,
            weight_decay=0.0078,
            map_location=device,
        )
        model = model.to(device)
        model.eval()
        
        # Process each target (for OOD: multiple targets; for ID: single target)
        for t_idx, (t_name, t_dir, log_name) in enumerate(zip(current_targets, current_dirs, log_names)):
            #load all samples into this list
            all_samples = []
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
            
            # HARDCODED: For OOD, process all batches; for ID, limit to max_batches
            batches_limit = float('inf') if ood_data else max_batches
            
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
                
                batch_count = 0
                for n, batch in enumerate(loader):
                    # Stop after batches_limit (inf for OOD, max_batches for ID)
                    if batch_count >= batches_limit:
                        break
                    
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
                    
                    batch_count += 1
                    if batch_count % 100 == 0:
                        print(f"  Batch {batch_count} done")
                
                # Average across dataset
                edge_connection_dict_all = {
                    key: value / batch_count
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
                
                os.makedirs(save_path_fold, exist_ok=True)
                
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
                
                print(f"  Baseline processing complete for {t_name} ({batch_count} batches)")
            
            # ================================================================
            # MC DROPOUT MODE: Instance-Level Stochastic Explanations
            # ================================================================
   
            else:        
                save_path_fold = os.path.join(save_dir_att, f"fold_{i}")
                os.makedirs(save_path_fold, exist_ok=True)
                
                sample_counter = 0
                for batch_idx, batch in enumerate(loader):
                    if sample_counter >= batches_limit: break
                    batch = batch.to(device)
                    
                    # 1. Run Explainer EXACTLY ONCE
                    model.eval()
                    explanation = explainer(
                        x=batch.x, edge_index=batch.edge_index, 
                        target=batch.y, pyg_batch=batch.batch
                    )
                    edge_mask_base = explanation.edge_mask.detach().cpu().numpy()
                    edge_index = explanation.edge_index.detach().cpu().numpy()
                    
                    # 2. Run Model 50 times for Predictive Uncertainty
                    model.train()
                    for m in model.modules():
                        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch_geometric.nn.norm.BatchNorm):
                            m.eval()
                    
                    all_preds = []
                    with torch.no_grad():
                        for t in range(50):
                            out = model(batch.x, batch.edge_index, batch.batch)
                            preds = torch.nn.functional.softmax(out, dim=1)
                            all_preds.append(preds.detach().cpu())
                    
                    all_preds = torch.stack(all_preds, dim=0).squeeze(1) # Shape: [50, 3]
                    mean_probs = all_preds.mean(dim=0)
                    pred_label_mode = mean_probs.argmax(dim=0).item()
                    
                    # 3. Calculate Entropies
                    predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10)).item()
                    entropies = -torch.sum(all_preds * torch.log(all_preds + 1e-10), dim=1)
                    aleatoric_entropy = entropies.mean().item()
                    epistemic_entropy = predictive_entropy - aleatoric_entropy
                    
                    # 4. Save
                    sample_data = {
                        "sample_id": f"{fold}_{t_name}_{batch_idx}",
                        "true_label": batch.y.item(),
                        "pred_label_mode": pred_label_mode,
                        "edge_mask_base": edge_mask_base,
                        "edge_index": edge_index,
                        "mean_probs": mean_probs.numpy(),
                        "predictive_entropy": predictive_entropy,
                        "aleatoric_entropy": aleatoric_entropy,
                        "epistemic_entropy": epistemic_entropy
                    }
                    
                    all_samples.append(sample_data)
                    
                    sample_counter += 1
                    if sample_counter % 100 == 0:
                        print(f"MC Sample {sample_counter}/{len(loader)} done")
                
                print(f"MC processing complete: {sample_counter} samples for {t_name}")

                file_prefix = f"{t_name}" if ood_data else "test_data"
                file_path = os.path.join(save_path_fold, f"{file_prefix}.json")
                with open(file_path, 'w') as f:
                    json.dump(all_samples, f)
                    
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
    parser.add_argument(
        "--max_batches",
        type=int,
        default=1000,
        help="Maximum number of batches to process for ID data (default: 1000). Ignored for OOD data."
    )
    args = parser.parse_args()
    compute_attention_explanations(args)