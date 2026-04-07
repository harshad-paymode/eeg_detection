# import torch
# from torch_geometric.loader import DataLoader
# import numpy as np
# from src.models import GATv2Lightning
# from src.utils.dataloader_utils import GraphDataset
# import lightning.pytorch as pl
# import os
# import json
# import types
# import torch_geometric
# from torchmetrics.classification import MulticlassCalibrationError
# from argparse import ArgumentParser
# from statistics import mean, stdev
# import wandb
# import logging
# from torch_geometric import seed_everything
# seed_everything(42)

# logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# api_key_file = open("/kaggle/working/eeg_detection/src/wandb_api_key.txt", "r")
# API_KEY = api_key_file.read()
# api_key_file.close()
# os.environ["WANDB_API_KEY"] = API_KEY

# parser = ArgumentParser()
# parser.add_argument("--checkpoint_dir", type=str, default="saved_models/")
# parser.add_argument("--test_data_dir", type=str, default="test_data/")
# parser.add_argument("--save_dir_metrics", type=str, default="save_unc_metrics/")
# parser.add_argument("--mc_dropout", action="store_true", default=False)
# # parser.add_argument("--temperature_path", type=str, default="temperatures/")
# parser.add_argument("--ood_data", action="store_true", default=False)
# args = parser.parse_args()

# CHECKPOINT_DIR = args.checkpoint_dir
# TEST_DATA_DIR = args.test_data_dir
# SAVE_DIR_METRICS = args.save_dir_metrics
# # TEMPERATURES_PATH = args.temperature_path
# OOD_DATA = args.ood_data

# INITIAL_CONFIG = dict(
#     mc_dropout=args.mc_dropout,
#     n_gat_layers=1,
#     hidden_dim=32,
#     slope=0.0025,
#     pooling_method="mean",
#     norm_method="batch",
#     activation="leaky_relu",
#     n_heads=9,
#     lr=0.0012,
#     weight_decay=0.0078
# )


# def compute_aurc(probs, targets):
#     """Computes Area Under the Risk-Coverage (AURC) directly on GPU."""
#     confidences, preds = torch.max(probs, dim=1)
#     errors = (preds != targets).float()
    
#     # Sort by confidence descending
#     sorted_indices = torch.argsort(confidences, descending=True)
#     sorted_errors = errors[sorted_indices]
    
#     # Calculate cumulative risk
#     n_samples = len(targets)
#     risks = torch.cumsum(sorted_errors, dim=0) / torch.arange(1, n_samples + 1, device=probs.device).float()
    
#     # Area under curve calculation
#     aurc = torch.sum(risks) / n_samples
#     return aurc.item()


# def compute_nll(probs, targets, num_classes=3):
#     """Computes Negative Log-Likelihood (NLL)."""
#     # probs shape: (N, num_classes)
#     # targets shape: (N,)
#     log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
#     nll = -log_probs[torch.arange(len(targets)), targets].mean()
#     return nll.item()


# def compute_total_entropy(probs):
#     """Computes Predictive Entropy (Total Entropy) per sample and mean."""
#     # probs shape: (N, num_classes)
#     entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # (N,)
#     return entropy


# def compute_aleatoric_entropy(all_probs_list):
#     """
#     Computes Aleatoric Entropy (Expected Entropy) from MC dropout samples.
    
#     Args:
#         all_probs_list: List of (N, num_classes) tensors from MC passes
    
#     Returns:
#         Aleatoric entropy per sample, shape (N,)
#     """
#     # Stack to (num_passes, N, num_classes)
#     stacked = torch.stack(all_probs_list, dim=0)
    
#     # Compute entropy for each pass, then average
#     # entropy shape per pass: (N,)
#     entropies = -torch.sum(stacked * torch.log(stacked + 1e-10), dim=2)  # (num_passes, N)
#     aleatoric = entropies.mean(dim=0)  # (N,)
    
#     return aleatoric


# def compute_mutual_information(all_probs_list, total_entropy):
#     """
#     Computes Mutual Information (Epistemic Uncertainty).
#     MI = Total Entropy - Aleatoric Entropy
    
#     Args:
#         all_probs_list: List of (N, num_classes) tensors from MC passes
#         total_entropy: Per-sample total entropy, shape (N,)
    
#     Returns:
#         Mutual information per sample, shape (N,)
#     """
#     aleatoric = compute_aleatoric_entropy(all_probs_list).to(total_entropy.device)
#     epistemic = total_entropy - aleatoric
#     return epistemic


# def compute_uncertainty_metrics():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     if INITIAL_CONFIG['mc_dropout'] and OOD_DATA:
#         project_name = "mc_uncertainty_eval_ood"
#     elif INITIAL_CONFIG['mc_dropout']:
#         project_name = "mc_uncertainty_eval"
#     elif OOD_DATA:
#         project_name = "base_uncertainty_eval_ood"
#     else:
#         project_name = "base_uncertainty_eval"

#     os.makedirs(SAVE_DIR_METRICS, exist_ok=True)
    
#     fold_list = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("fold_")]
#     fold_list.sort()
    
#     # if INITIAL_CONFIG['mc_dropout']:
#     #     temp_file_path = os.path.join(TEMPERATURES_PATH, "optimal_temperatures.json")
#     #     with open(temp_file_path, "r") as f:
#     #         optimal_temperatures = json.load(f)
#     #     print(f"Loaded optimal temperatures from {temp_file_path}")

#     # Determine targets once
#     if OOD_DATA:
#         target_names = sorted([p for p in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, p))])
#     else:
#         target_names = None

#     # Dictionary to hold cross-fold metrics per target
#     summary_metrics = {}

#     for n, fold in enumerate(fold_list):
#         print(f"Evaluating Fold {n} | MC Dropout: {INITIAL_CONFIG['mc_dropout']}")
        
#         # 1. Determine exact test targets for this fold
#         if OOD_DATA:
#             current_targets = target_names
#             current_dirs = [os.path.join(TEST_DATA_DIR, p) for p in current_targets]
#             log_names = [f"{fold}_{p}" for p in current_targets]
#         else:
#             current_targets = ["id_test"]
#             current_dirs = [os.path.join(TEST_DATA_DIR, fold)]
#             log_names = [f"fold_{fold}"]

#         checkpoint_fold_dir = os.path.join(CHECKPOINT_DIR, fold)
#         checkpoint_path = os.path.join(checkpoint_fold_dir, os.listdir(checkpoint_fold_dir)[0])
        
#         # Grab features shape dynamically
#         features_shape = GraphDataset(current_dirs[0])[0].x.shape[-1]
        
#         model = GATv2Lightning.load_from_checkpoint(
#             checkpoint_path,
#             in_features=features_shape,
#             n_classes=3,
#             n_gat_layers=INITIAL_CONFIG['n_gat_layers'],
#             hidden_dim=INITIAL_CONFIG['hidden_dim'],
#             n_heads=INITIAL_CONFIG['n_heads'],
#             slope=INITIAL_CONFIG['slope'],
#             dropout_on=INITIAL_CONFIG['mc_dropout'],
#             pooling_method=INITIAL_CONFIG['pooling_method'],
#             activation=INITIAL_CONFIG['activation'],
#             norm_method=INITIAL_CONFIG['norm_method'],
#             lr=INITIAL_CONFIG['lr'],
#             weight_decay=INITIAL_CONFIG['weight_decay'],
#             map_location=device,
#         )
        
#         # if INITIAL_CONFIG['mc_dropout']:
#         #     model.temperature = optimal_temperatures[fold]

#         wandb_logger = pl.loggers.WandbLogger(log_model=False)
#         trainer = pl.Trainer(
#             accelerator="auto", max_epochs=1, devices=1, enable_progress_bar=False,
#             deterministic=False, logger=wandb_logger, enable_model_summary=False,
#         )

#         dir_fold = os.path.join(SAVE_DIR_METRICS, f"fold_{n}")
#         os.makedirs(dir_fold, exist_ok=True)

#         # Evaluate Each Target
#         for t_name, t_dir, log_name in zip(current_targets, current_dirs, log_names):
#             wandb.init(project=project_name, name=log_name, config=INITIAL_CONFIG)
            
#             ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=15, norm='l1').to(device)

#             dataset = GraphDataset(t_dir)
#             loader = DataLoader(dataset, batch_size=1024, shuffle=False)

#             # Set exactly the modes we want
#             if INITIAL_CONFIG['mc_dropout']:
#                 model.train()
#                 for m in model.modules():
#                     if isinstance(m, (torch.nn.BatchNorm1d, torch_geometric.nn.norm.BatchNorm)):
#                         m.eval()
#             else:
#                 model.eval()

#             all_preds = []
            
#             # Manual inference loop (Bypasses Lightning completely)
#             with torch.no_grad():
#                 for p in range(50 if INITIAL_CONFIG['mc_dropout'] else 1):
#                     pass_preds = []
#                     for batch in loader:
#                         batch = batch.to(device)
#                         out = model(batch.x, batch.edge_index, batch.batch)
#                         pass_preds.append(out.detach().cpu())
                        
#                     preds = torch.cat(pass_preds, dim=0)
                    
#                     if INITIAL_CONFIG['mc_dropout']:
#                         preds = torch.nn.functional.softmax(preds, dim=1)
                    
#                     all_preds.append(preds)

#             # Aggregate
#             if INITIAL_CONFIG['mc_dropout']:
#                 preds_raw = torch.stack(all_preds).mean(dim=0)
#                 preds = preds_raw.argmax(dim=1)
#             else:
#                 preds_raw = all_preds[0]
#                 preds = torch.nn.functional.softmax(preds_raw, dim=1).argmax(dim=1)

#             preds_raw, preds = preds_raw.to(device), preds.to(device)
#             ground_truth = torch.tensor([data.y.int().item() for data in dataset]).to(device)

#             # =========================================================
#             # COMPUTE UNCERTAINTY METRICS
#             # =========================================================
            
#             # ECE (Calibration)
#             fold_ece = ece_metric(preds_raw, ground_truth).item()
            
#             # NLL
#             fold_nll = compute_nll(preds_raw, ground_truth, num_classes=3)
            
#             # Total Entropy (Predictive Entropy)
#             total_entropy_per_sample = compute_total_entropy(preds_raw)
#             fold_total_entropy = total_entropy_per_sample.mean().item()
            
#             # Aleatoric Entropy (Expected Entropy) - for MC Dropout
#             if INITIAL_CONFIG['mc_dropout']:
#                 aleatoric_entropy_per_sample = compute_aleatoric_entropy(all_preds)
#                 fold_aleatoric_entropy = aleatoric_entropy_per_sample.mean().item()
                
#                 # Epistemic (Mutual Information)
#                 epistemic_entropy_per_sample = compute_mutual_information(all_preds, total_entropy_per_sample)
#                 fold_epistemic_entropy = epistemic_entropy_per_sample.mean().item()
#             else:
#                 # For non-MC dropout, aleatoric = total, epistemic = 0
#                 aleatoric_entropy_per_sample = total_entropy_per_sample
#                 epistemic_entropy_per_sample = torch.zeros_like(total_entropy_per_sample)
#                 fold_aleatoric_entropy = fold_total_entropy
#                 fold_epistemic_entropy = 0.0
            
#             # AURC
#             fold_aurc = compute_aurc(preds_raw, ground_truth)
            
#             # Log to W&B
#             wandb.log({
#                 "ECE": fold_ece,
#                 "NLL": fold_nll,
#                 "Total_Entropy": fold_total_entropy,
#                 "Aleatoric_Entropy": fold_aleatoric_entropy,
#                 "Epistemic_Entropy": fold_epistemic_entropy,
#                 "AURC": fold_aurc,
#             })
#             wandb.finish()

#             # Save fold-level scalar results
#             fold_results = {
#                 "fold": fold,
#                 "target": t_name,
#                 "ECE": fold_ece,
#                 "NLL": fold_nll,
#                 "Total_Entropy": fold_total_entropy,
#                 "Aleatoric_Entropy": fold_aleatoric_entropy,
#                 "Epistemic_Entropy": fold_epistemic_entropy,
#                 "AURC": fold_aurc,
#             }
            
#             file_prefix = f"{t_name}_" if OOD_DATA else ""
#             with open(os.path.join(dir_fold, f"{file_prefix}unc_results.json"), "w") as f:
#                 json.dump(fold_results, f)

#             # Calucating predictions and correct/incorrect flag
#             preds_class = preds_raw.argmax(dim=1)
#             is_correct = (preds_class == ground_truth)

#             # Save per-sample data for later plotting
#             per_sample_data = {
#                 "sample_id": np.array([f"{fold}_{t_name}_{i}" for i in range(len(ground_truth))]), # Patient + Index
#                 "mean_probs": preds_raw.cpu().numpy(),
#                 "pred_labels": preds_class.cpu().numpy(),
#                 "true_labels": ground_truth.cpu().numpy(),
#                 "confidence": preds_raw.max(dim=1).values.cpu().numpy(),
#                 "is_correct": is_correct.cpu().numpy(),
#                 "total_entropy": total_entropy_per_sample.cpu().numpy(),
#                 "aleatoric_entropy": aleatoric_entropy_per_sample.cpu().numpy(),
#                 "epistemic_entropy": epistemic_entropy_per_sample.cpu().numpy(),
#             }
            
#             npz_path = os.path.join(dir_fold, f"{file_prefix}per_sample_data.npz")
#             np.savez(npz_path, **per_sample_data)
            
#             # Accumulate for cross-fold summary
#             if t_name not in summary_metrics:
#                 summary_metrics[t_name] = {
#                     "ece": [],
#                     "nll": [],
#                     "total_entropy": [],
#                     "aleatoric_entropy": [],
#                     "epistemic_entropy": [],
#                     "aurc": [],
#                 }
            
#             summary_metrics[t_name]["ece"].append(fold_ece)
#             summary_metrics[t_name]["nll"].append(fold_nll)
#             summary_metrics[t_name]["total_entropy"].append(fold_total_entropy)
#             summary_metrics[t_name]["aleatoric_entropy"].append(fold_aleatoric_entropy)
#             summary_metrics[t_name]["epistemic_entropy"].append(fold_epistemic_entropy)
#             summary_metrics[t_name]["aurc"].append(fold_aurc)

#     # =============================================================
#     # FINAL SUMMARY LOGGING (per target if OOD, once if ID)
#     # =============================================================
#     for t_name, metrics in summary_metrics.items():
#         log_name = f"summary_uncertainty_{t_name}" if OOD_DATA else "summary_uncertainty"
#         wandb.init(project=project_name, name=log_name)
        
#         summary_results = {
#             "final_mean_ECE": mean(metrics["ece"]),
#             "final_ECE_std": stdev(metrics["ece"]) if len(metrics["ece"]) > 1 else 0.0,
            
#             "final_mean_NLL": mean(metrics["nll"]),
#             "final_NLL_std": stdev(metrics["nll"]) if len(metrics["nll"]) > 1 else 0.0,
            
#             "final_mean_Total_Entropy": mean(metrics["total_entropy"]),
#             "final_Total_Entropy_std": stdev(metrics["total_entropy"]) if len(metrics["total_entropy"]) > 1 else 0.0,
            
#             "final_mean_Aleatoric_Entropy": mean(metrics["aleatoric_entropy"]),
#             "final_Aleatoric_Entropy_std": stdev(metrics["aleatoric_entropy"]) if len(metrics["aleatoric_entropy"]) > 1 else 0.0,
            
#             "final_mean_Epistemic_Entropy": mean(metrics["epistemic_entropy"]),
#             "final_Epistemic_Entropy_std": stdev(metrics["epistemic_entropy"]) if len(metrics["epistemic_entropy"]) > 1 else 0.0,
            
#             "final_mean_AURC": mean(metrics["aurc"]),
#             "final_AURC_std": stdev(metrics["aurc"]) if len(metrics["aurc"]) > 1 else 0.0,
#         }
        
#         wandb.log(summary_results)
#         wandb.finish()
        
#         file_suffix = f"_{t_name}" if OOD_DATA else ""
#         with open(os.path.join(SAVE_DIR_METRICS, f"summary_unc_results{file_suffix}.json"), "w") as f:
#             json.dump(summary_results, f)


# if __name__ == "__main__":
#     compute_uncertainty_metrics()

import torch
from torch_geometric.loader import DataLoader
import numpy as np
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import torch_geometric
import json
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.classification import MulticlassCalibrationError
from argparse import ArgumentParser
from statistics import mean, stdev
import wandb
import logging
from torch_geometric import seed_everything
seed_everything(42)

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

api_key_file = open("/kaggle/working/eeg_detection/src/wandb_api_key.txt", "r")
API_KEY = api_key_file.read()
api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY

parser = ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default="saved_models/")
parser.add_argument("--test_data_dir", type=str, default="test_data/")
parser.add_argument("--save_dir_metrics", type=str, default="save_unc_metrics/")
parser.add_argument("--mc_dropout", action="store_true", default=False)
parser.add_argument("--ood_data", action="store_true", default=False)
args = parser.parse_args()

CHECKPOINT_DIR = args.checkpoint_dir
TEST_DATA_DIR = args.test_data_dir
SAVE_DIR_METRICS = args.save_dir_metrics
OOD_DATA = args.ood_data

INITIAL_CONFIG = dict(
    mc_dropout=args.mc_dropout,
    n_gat_layers=1,
    hidden_dim=32,
    slope=0.0025,
    pooling_method="mean",
    norm_method="batch",
    activation="leaky_relu",
    n_heads=9,
    lr=0.0012,
    weight_decay=0.0078
)

def compute_prediction_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR_METRICS, exist_ok=True)
    
    fold_list = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("fold_")]
    fold_list.sort()

    # =========================================================================
    # OOD DATA: ENSEMBLE EVALUATION
    # =========================================================================
    if OOD_DATA:
        project_name = "ensemble_mc_unc_eval_ood" if INITIAL_CONFIG['mc_dropout'] else "ensemble_base_unc_eval_ood"
        target_names = sorted([p for p in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, p))])
        
        for t_name in target_names:
            print(f"\n--- Evaluating OOD Patient: {t_name} (Ensemble over 10 Folds) ---")
            t_dir = os.path.join(TEST_DATA_DIR, t_name)
            
            dataset = GraphDataset(t_dir)
            loader = DataLoader(dataset, batch_size=1024, shuffle=False)
            ground_truth = torch.tensor([data.y.int().item() for data in dataset])
            
            all_ensemble_passes = []
            
            for n, fold in enumerate(fold_list):
                print(f"  -> Gathering predictions from {fold}...")
                checkpoint_fold_dir = os.path.join(CHECKPOINT_DIR, fold)
                checkpoint_path = os.path.join(checkpoint_fold_dir, os.listdir(checkpoint_fold_dir)[0])
                features_shape = dataset[0].x.shape[-1]
                
                model = GATv2Lightning.load_from_checkpoint(
                    checkpoint_path, in_features=features_shape, n_classes=3,
                    n_gat_layers=INITIAL_CONFIG['n_gat_layers'], hidden_dim=INITIAL_CONFIG['hidden_dim'],
                    n_heads=INITIAL_CONFIG['n_heads'], slope=INITIAL_CONFIG['slope'],
                    dropout_on=INITIAL_CONFIG['mc_dropout'], pooling_method=INITIAL_CONFIG['pooling_method'],
                    activation=INITIAL_CONFIG['activation'], norm_method=INITIAL_CONFIG['norm_method'],
                    lr=INITIAL_CONFIG['lr'], weight_decay=INITIAL_CONFIG['weight_decay'], map_location=device
                )
                
                if INITIAL_CONFIG['mc_dropout']:
                    model.train()
                    for m in model.modules():
                        if isinstance(m, (torch.nn.BatchNorm1d, torch_geometric.nn.norm.BatchNorm)):
                            m.eval()
                else:
                    model.eval()

                with torch.no_grad():
                    num_passes = 50 if INITIAL_CONFIG['mc_dropout'] else 1
                    for p in range(num_passes):
                        pass_preds = []
                        for batch in loader:
                            batch = batch.to(device)
                            out = model(batch.x, batch.edge_index, batch.batch)
                            pass_preds.append(out.detach().cpu())
                        
                        preds_logits = torch.cat(pass_preds, dim=0) # [Total_Samples, 3]
                        # Apply Softmax immediately so we ensemble probabilities, not raw logits
                        probs = torch.nn.functional.softmax(preds_logits, dim=1) 
                        all_ensemble_passes.append(probs)

            # --- Ensemble Metrics Calculation ---
            stacked = torch.stack(all_ensemble_passes) # [Num_Folds * Num_Passes, Total_Samples, 3]
            print(f"Total passes ensembled for {t_name}: {stacked.shape[0]}")
            
            mean_probs = stacked.mean(dim=0)
            preds = mean_probs.argmax(dim=1)
            
            confidences, _ = mean_probs.max(dim=1)
            is_correct = (preds == ground_truth).numpy()
            confidences_np = confidences.numpy()
            
            # Entropy Calculations (Dim consistency applied)
            predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
            aleatoric_entropy = -torch.sum(stacked * torch.log(stacked + 1e-10), dim=2).mean(dim=0)
            epistemic_entropy = torch.clamp(predictive_entropy - aleatoric_entropy, min=0.0)

            # System Level Metrics
            ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='l1')
            ece = ece_metric(mean_probs, ground_truth).item()
            nll = torch.nn.functional.nll_loss(torch.log(mean_probs + 1e-10), ground_truth.long()).item()

            sort_indices = np.argsort(-confidences_np)
            sorted_is_correct = is_correct[sort_indices]
            errors = 1 - sorted_is_correct
            cumulative_errors = np.cumsum(errors)
            risk = cumulative_errors / np.arange(1, len(errors) + 1)
            aurc = np.mean(risk)

            results = {
                "patient": t_name,
                "total_ensemble_passes": stacked.shape[0],
                "ECE": ece,
                "NLL": nll,
                "AURC": aurc,
                "mean_Total_Entropy": predictive_entropy.mean().item(),
                "mean_Aleatoric_Entropy": aleatoric_entropy.mean().item(),
                "mean_Epistemic_Entropy": epistemic_entropy.mean().item()
            }
            
            wandb.init(project=project_name, name=f"ensemble_{t_name}", config=INITIAL_CONFIG)
            wandb.log(results)
            wandb.finish()

            # Save Output
            dir_save = os.path.join(SAVE_DIR_METRICS, f"ensemble_{t_name}")
            os.makedirs(dir_save, exist_ok=True)
            
            np.savez(
                os.path.join(dir_save, f"{t_name}_per_sample_data.npz"),
                confidence=confidences_np,
                is_correct=is_correct,
                true_labels=ground_truth.numpy(),
                pred_labels=preds.numpy(),
                predictive_entropy=predictive_entropy.numpy(),
                aleatoric_entropy=aleatoric_entropy.numpy(),
                epistemic_entropy=epistemic_entropy.numpy()
            )
            with open(os.path.join(dir_save, f"{t_name}_unc_results.json"), "w") as f:
                json.dump(results, f, indent=4)


    # =========================================================================
    # ID DATA: STANDARD FOLD-WISE EVALUATION
    # =========================================================================
    else:
        project_name = "mc_dropout_unc_eval" if INITIAL_CONFIG['mc_dropout'] else "base_unc_eval"
        summary_metrics = {"ece": [], "nll": [], "tot_ent": [], "al_ent": [], "ep_ent": [], "aurc": []}

        for n, fold in enumerate(fold_list):
            print(f"Evaluating ID Data | Fold {n}")
            t_dir = os.path.join(TEST_DATA_DIR, fold)
            checkpoint_fold_dir = os.path.join(CHECKPOINT_DIR, fold)
            checkpoint_path = os.path.join(checkpoint_fold_dir, os.listdir(checkpoint_fold_dir)[0])
            
            dataset = GraphDataset(t_dir)
            loader = DataLoader(dataset, batch_size=1024, shuffle=False)
            features_shape = dataset[0].x.shape[-1]
            
            model = GATv2Lightning.load_from_checkpoint(
                checkpoint_path, in_features=features_shape, n_classes=3,
                n_gat_layers=INITIAL_CONFIG['n_gat_layers'], hidden_dim=INITIAL_CONFIG['hidden_dim'],
                n_heads=INITIAL_CONFIG['n_heads'], slope=INITIAL_CONFIG['slope'],
                dropout_on=INITIAL_CONFIG['mc_dropout'], pooling_method=INITIAL_CONFIG['pooling_method'],
                activation=INITIAL_CONFIG['activation'], norm_method=INITIAL_CONFIG['norm_method'],
                lr=INITIAL_CONFIG['lr'], weight_decay=INITIAL_CONFIG['weight_decay'], map_location=device
            )
            
            if INITIAL_CONFIG['mc_dropout']:
                model.train()
                for m in model.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch_geometric.nn.norm.BatchNorm)):
                        m.eval()
            else:
                model.eval()

            wandb.init(project=project_name, name=f"fold_{fold}", config=INITIAL_CONFIG)

            all_preds = []
            with torch.no_grad():
                for p in range(50 if INITIAL_CONFIG['mc_dropout'] else 1):
                    pass_preds = []
                    for batch in loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        pass_preds.append(out.detach().cpu())
                    
                    preds = torch.cat(pass_preds, dim=0)
                    probs = torch.nn.functional.softmax(preds, dim=1)
                    all_preds.append(probs)

            stacked = torch.stack(all_preds)
            mean_probs = stacked.mean(dim=0)
            preds = mean_probs.argmax(dim=1)
            ground_truth = torch.tensor([data.y.int().item() for data in dataset])

            confidences, _ = mean_probs.max(dim=1)
            is_correct = (preds == ground_truth).numpy()
            
            predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
            aleatoric_entropy = -torch.sum(stacked * torch.log(stacked + 1e-10), dim=2).mean(dim=0)
            epistemic_entropy = torch.clamp(predictive_entropy - aleatoric_entropy, min=0.0)

            ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='l1')
            ece = ece_metric(mean_probs, ground_truth).item()
            nll = torch.nn.functional.nll_loss(torch.log(mean_probs + 1e-10), ground_truth.long()).item()

            sort_indices = np.argsort(-confidences.numpy())
            sorted_is_correct = is_correct[sort_indices]
            errors = 1 - sorted_is_correct
            risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
            aurc = np.mean(risk)

            fold_results = {
                "fold": fold, "ECE": ece, "NLL": nll, "AURC": aurc,
                "mean_Total_Entropy": predictive_entropy.mean().item(),
                "mean_Aleatoric_Entropy": aleatoric_entropy.mean().item(),
                "mean_Epistemic_Entropy": epistemic_entropy.mean().item()
            }
            wandb.log(fold_results)
            wandb.finish()

            dir_fold = os.path.join(SAVE_DIR_METRICS, f"fold_{n}")
            os.makedirs(dir_fold, exist_ok=True)
            
            np.savez(
                os.path.join(dir_fold, "per_sample_data.npz"),
                confidence=confidences.numpy(), is_correct=is_correct, true_labels=ground_truth.numpy(),
                pred_labels=preds.numpy(), predictive_entropy=predictive_entropy.numpy(),
                aleatoric_entropy=aleatoric_entropy.numpy(), epistemic_entropy=epistemic_entropy.numpy()
            )
            with open(os.path.join(dir_fold, "unc_results.json"), "w") as f:
                json.dump(fold_results, f, indent=4)
            
            summary_metrics["ece"].append(ece)
            summary_metrics["nll"].append(nll)
            summary_metrics["aurc"].append(aurc)
            summary_metrics["tot_ent"].append(predictive_entropy.mean().item())
            summary_metrics["al_ent"].append(aleatoric_entropy.mean().item())
            summary_metrics["ep_ent"].append(epistemic_entropy.mean().item())

        # ID Summary Log
        wandb.init(project=project_name, name="summary_uncertainty_metrics")
        summary_results = {
            "final_mean_ECE": mean(summary_metrics["ece"]), "final_ECE_std": stdev(summary_metrics["ece"]),
            "final_mean_NLL": mean(summary_metrics["nll"]), "final_NLL_std": stdev(summary_metrics["nll"]),
            "final_mean_AURC": mean(summary_metrics["aurc"]), "final_AURC_std": stdev(summary_metrics["aurc"]),
            "final_mean_Total_Entropy": mean(summary_metrics["tot_ent"]), "final_Total_Entropy_std": stdev(summary_metrics["tot_ent"]),
            "final_mean_Aleatoric_Entropy": mean(summary_metrics["al_ent"]), "final_Aleatoric_Entropy_std": stdev(summary_metrics["al_ent"]),
            "final_mean_Epistemic_Entropy": mean(summary_metrics["ep_ent"]), "final_Epistemic_Entropy_std": stdev(summary_metrics["ep_ent"])
        }
        wandb.log(summary_results)
        wandb.finish()
        with open(os.path.join(SAVE_DIR_METRICS, "summary_unc_results.json"), "w") as f:
            json.dump(summary_results, f, indent=4)

if __name__ == "__main__":
    compute_prediction_metrics()