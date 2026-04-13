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
import logging
from argparse import ArgumentParser
from torch_geometric import seed_everything
seed_everything(42)

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def compute_attention_explanations(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_att_base = args.save_dir_att
    mc_dropout = args.mc_dropout
    ood_data = args.ood_data
    
    # Determine final save directory
    save_dir_att = save_dir_att_base

    os.makedirs(save_dir_att, exist_ok=True)
    
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
        print(f"Fold {fold} (Index {i}) | MC Dropout: {mc_dropout} | OOD: {ood_data}")
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
            dropout_on=False,
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
                num_workers=0,          # FIX: Prevent IPC deadlocks
                prefetch_factor=None,   # FIX: Must be None if num_workers=0
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
                
                batch_count = 0
                for n, batch in enumerate(loader):
                    batch = batch.to(device)
                    explanation = explainer(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        target=batch.y,
                        pyg_batch=batch.batch,
                    )
                    
                    # FIX: Safely extract scalars
                    prediction = torch.argmax(explanation.prediction).item()
                    batch_y = batch.y.item()
                    
                    for edge_idx in range(explanation.edge_index.size(1)):
                        edge = explanation.edge_index[:, edge_idx].tolist()
                        edge.sort()
                        edge = str(tuple(edge))
                        edge_mask = explanation.edge_mask[edge_idx].item()
                        
                        if edge in edge_connection_dict_all.keys():
                            edge_connection_dict_all[edge] += edge_mask
                        else:
                            edge_connection_dict_all[edge] = edge_mask
                        
                        if batch_y == 0 and prediction == 0:
                            if edge in edge_connection_dict_preictal.keys():
                                edge_connection_dict_preictal[edge] += edge_mask
                            else:
                                edge_connection_dict_preictal[edge] = edge_mask
                        elif batch_y == 1 and prediction == 1:
                            if edge in edge_connection_dict_ictal.keys():
                                edge_connection_dict_ictal[edge] += edge_mask
                            else:
                                edge_connection_dict_ictal[edge] = edge_mask
                        elif batch_y == 2 and prediction == 2:
                            if edge in edge_connection_dict_interictal.keys():
                                edge_connection_dict_interictal[edge] += edge_mask
                            else:
                                edge_connection_dict_interictal[edge] = edge_mask
                    
                    if batch_y == 0 and prediction == 0:
                        preictal_cntr += 1
                    elif batch_y == 1 and prediction == 1:
                        ictal_cntr += 1
                    elif batch_y == 2 and prediction == 2:
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
            # MC DROPOUT MODE: Save mask for every sample seperately
            # 1. Here only the masks are calculated for MC dropout
            # 2. The predictions will be cross referenced from compute_uncertainty_matrix results using sample id
            # ================================================================
   
            else:        
                save_path_fold = os.path.join(save_dir_att, f"fold_{i}")
                os.makedirs(save_path_fold, exist_ok=True)
                
                sample_counter = 0
                for batch_idx, batch in enumerate(loader):
                    batch = batch.to(device)
                    
                    # 1. Run Explainer EXACTLY ONCE
                    explanation = explainer(
                        x=batch.x, edge_index=batch.edge_index, 
                        target=batch.y, pyg_batch=batch.batch
                    )
                    
                    # FIX: Extract edge_index so downstream plotting script knows which edges to draw
                    edge_mask_base = explanation.edge_mask.detach().cpu().numpy()
                    edge_index = explanation.edge_index.detach().cpu().numpy()
                    
                    # Save true_label and edge_index for downstream parsing
                    sample_data = {
                        "sample_id": f"{fold}_{t_name}_{batch_idx}",
                        "true_label": batch.y.item(),
                        "prediction": torch.argmax(explanation.prediction).item(),
                        "edge_index": edge_index.tolist(),
                        "edge_mask_base": edge_mask_base.tolist(),
                    }
                    
                    all_samples.append(sample_data)
                    
                    sample_counter += 1
                    if sample_counter % 100 == 0:
                        print(f"MC Sample {sample_counter}/{len(loader)} done")

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
    args = parser.parse_args()
    compute_attention_explanations(args)