import torch
import time
import json
import math
import numpy as np
import os
from torch_geometric.loader import DataLoader
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
from torch_geometric.explain import GNNExplainer, AttentionExplainer, Explainer, ModelConfig
from argparse import ArgumentParser
import logging
from torch_geometric import seed_everything
seed_everything(42)

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

parser = ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--drop_fraction", type=float, default=0.20, help="Fraction to drop")
parser.add_argument("--max_samples_per_fold", type=int, default=200)
parser.add_argument("--save_results", type=str, default="save_ablation_results/")
parser.add_argument("--ood_data", action="store_true", default=False)
args = parser.parse_args()

def compute_sparsity(mask):
    """Calculates normalized sparsity (1 = extremely focused/sparse, 0 = uniform)."""
    if mask.sum() == 0: return 0.0
    p = mask / mask.sum()
    entropy = -torch.sum(p * torch.log(p + 1e-10))
    max_entropy = math.log(len(mask)) if len(mask) > 1 else 1.0
    return max(0.0, 1.0 - (entropy.item() / max_entropy))

def run_ablation():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_list = sorted([f for f in os.listdir(args.checkpoint_dir) if f.startswith("fold_")])

    # Enforce OOD constraints
    if args.ood_data:
        fold_list = [f for f in fold_list if f in ['fold_0', 'fold_1', 'fold_2', 'fold_3']]
        target_names = sorted([p for p in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, p))])
    else:
        target_names = None

    results = {}

    edge_config = ModelConfig("multiclass_classification", task_level="graph", return_type="raw")
    node_config = ModelConfig("multiclass_classification", task_level="graph", return_type="raw")

    for fold in fold_list:
        print(f"\n{'='*40}\nProcessing {fold} | OOD: {args.ood_data}\n{'='*40}")
        checkpoint_fold_dir = os.path.join(args.checkpoint_dir, fold)
        checkpoint_path = os.path.join(checkpoint_fold_dir, os.listdir(checkpoint_fold_dir)[0])

        # Target Setup
        if args.ood_data:
            current_targets = target_names
            current_dirs = [os.path.join(args.data_dir, t) for t in current_targets]
        else:
            current_targets = [fold]
            current_dirs = [os.path.join(args.data_dir, fold)]

        features_shape = GraphDataset(current_dirs[0])[0].x.shape[-1]
        model = GATv2Lightning.load_from_checkpoint(
            checkpoint_path, in_features=features_shape, n_classes=3,
            n_gat_layers=1, hidden_dim=32, n_heads=9, slope=0.0025,
            dropout=False, pooling_method="mean", activation="leaky_relu",
            norm_method="batch", lr=0.0012, weight_decay=0.0078, map_location=device
        ).to(device)
        model.eval()

        att_explainer = Explainer(model, algorithm=AttentionExplainer(), explanation_type="model", model_config=edge_config, edge_mask_type="object", node_mask_type=None)
        gnn_explainer = Explainer(model, algorithm=GNNExplainer(epochs=50, lr=0.01), explanation_type="model", model_config=node_config, node_mask_type="attributes", edge_mask_type=None)

        for t_name, t_dir in zip(current_targets, current_dirs):
            print(f"Target: {t_name}")
            if t_name not in results:
                results[t_name] = {
                    "attention_edges": {"time": [], "drop": [], "sparsity": []},
                    "random_edges": {"drop": []},
                    "gnn_nodes": {"time": [], "drop": [], "sparsity": []},
                    "random_nodes": {"drop": []}
                }

            dataset = GraphDataset(t_dir)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            samples_processed = 0

            for batch in loader:
                if samples_processed >= args.max_samples_per_fold: break
                batch = batch.to(device)
                
                with torch.no_grad():
                    base_out = model(batch.x, batch.edge_index, batch.batch)
                    base_probs = torch.nn.functional.softmax(base_out, dim=1)
                    pred_class = base_probs.argmax(dim=1).item()
                    base_conf = base_probs[0, pred_class].item()
                
                if pred_class != batch.y.item(): continue
                samples_processed += 1

                # ==========================================
                # DOMAIN 1: EDGE ABLATION
                # ==========================================
                num_edges = batch.edge_index.size(1)
                num_drop_edges = int(num_edges * args.drop_fraction)
                
                if num_drop_edges > 0:
                    t0 = time.time()
                    att_exp = att_explainer(x=batch.x, edge_index=batch.edge_index, target=batch.y, pyg_batch=batch.batch)
                    results[t_name]["attention_edges"]["time"].append(time.time() - t0)
                    
                    att_mask = att_exp.edge_mask.detach().cpu()
                    results[t_name]["attention_edges"]["sparsity"].append(compute_sparsity(att_mask))
                    
                    _, drop_idx_att = torch.topk(att_mask, num_drop_edges)
                    keep_edges_att = torch.ones(num_edges, dtype=torch.bool)
                    keep_edges_att[drop_idx_att] = False
                    with torch.no_grad():
                        out_att = model(batch.x, batch.edge_index[:, keep_edges_att.to(device)], batch.batch)
                        results[t_name]["attention_edges"]["drop"].append(base_conf - torch.nn.functional.softmax(out_att, dim=1)[0, pred_class].item())
                    
                    rand_edge_idx = torch.randperm(num_edges)[:num_drop_edges]
                    keep_edges_rand = torch.ones(num_edges, dtype=torch.bool)
                    keep_edges_rand[rand_edge_idx] = False
                    with torch.no_grad():
                        out_rand_e = model(batch.x, batch.edge_index[:, keep_edges_rand.to(device)], batch.batch)
                        results[t_name]["random_edges"]["drop"].append(base_conf - torch.nn.functional.softmax(out_rand_e, dim=1)[0, pred_class].item())

                # ==========================================
                # DOMAIN 2: NODE ABLATION
                # ==========================================
                num_nodes = batch.x.size(0)
                num_drop_nodes = int(num_nodes * args.drop_fraction)
                
                if num_drop_nodes > 0:
                    t0 = time.time()
                    gnn_exp = gnn_explainer(x=batch.x, edge_index=batch.edge_index, target=batch.y, pyg_batch=batch.batch)
                    results[t_name]["gnn_nodes"]["time"].append(time.time() - t0)
                    
                    node_mask = gnn_exp.node_mask.detach().cpu()
                    if node_mask.dim() > 1: node_mask = node_mask.mean(dim=1)
                    results[t_name]["gnn_nodes"]["sparsity"].append(compute_sparsity(node_mask))
                    
                    _, drop_idx_gnn = torch.topk(node_mask, num_drop_nodes)
                    ablated_x_gnn = batch.x.clone()
                    ablated_x_gnn[drop_idx_gnn.to(device)] = 0.0
                    with torch.no_grad():
                        out_gnn = model(ablated_x_gnn, batch.edge_index, batch.batch)
                        results[t_name]["gnn_nodes"]["drop"].append(base_conf - torch.nn.functional.softmax(out_gnn, dim=1)[0, pred_class].item())
                    
                    rand_node_idx = torch.randperm(num_nodes)[:num_drop_nodes]
                    ablated_x_rand = batch.x.clone()
                    ablated_x_rand[rand_node_idx.to(device)] = 0.0
                    with torch.no_grad():
                        out_rand_n = model(ablated_x_rand, batch.edge_index, batch.batch)
                        results[t_name]["random_nodes"]["drop"].append(base_conf - torch.nn.functional.softmax(out_rand_n, dim=1)[0, pred_class].item())

            print(f"Ablation Done For {t_name} and {fold}")
    # Compile Final JSON 
    final_summary = {}
    for t_name, res in results.items():
        final_summary[t_name] = {
            "attention_edges": {
                "time_per_sample": float(np.mean(res['attention_edges']['time'])),
                "sparsity": float(np.mean(res['attention_edges']['sparsity'])),
                "fidelity_drop": float(np.mean(res['attention_edges']['drop']))
            },
            "random_edges": {
                "fidelity_drop": float(np.mean(res['random_edges']['drop']))
            },
            "gnn_nodes": {
                "time_per_sample": float(np.mean(res['gnn_nodes']['time'])),
                "sparsity": float(np.mean(res['gnn_nodes']['sparsity'])),
                "fidelity_drop": float(np.mean(res['gnn_nodes']['drop']))
            },
            "random_nodes": {
                "fidelity_drop": float(np.mean(res['random_nodes']['drop']))
            }
        }

    os.makedirs(args.save_results, exist_ok=True)
    file_prefix = "OOD_" if args.ood_data else "ID_"
    save_path = os.path.join(args.save_results, f"{file_prefix}ablation_summary.json")
    
    with open(save_path, "w") as f:
        json.dump(final_summary, f, indent=4)
        
    print(f"\nDone! Results securely saved to: {save_path}")

if __name__ == "__main__":
    run_ablation()