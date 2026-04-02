import torch
from torch_geometric.loader import DataLoader
import numpy as np
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import json
import types
from torchmetrics.classification import MulticlassCalibrationError
from argparse import ArgumentParser
from statistics import mean, stdev
import wandb

api_key_file = open("/kaggle/working/eeg_detection/src/wandb_api_key.txt", "r")
API_KEY = api_key_file.read()
api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY

parser = ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default="saved_models/")
parser.add_argument("--test_data_dir", type=str, default="test_data/")
parser.add_argument("--save_dir_metrics", type=str, default="save_unc_metrics/")
parser.add_argument("--mc_dropout", action="store_true", default=False)
# parser.add_argument("--temperature_path", type=str, default="temperatures/")
parser.add_argument("--ood_data", action="store_true", default=False)
args = parser.parse_args()

CHECKPOINT_DIR = args.checkpoint_dir
TEST_DATA_DIR = args.test_data_dir
SAVE_DIR_METRICS = args.save_dir_metrics
# TEMPERATURES_PATH = args.temperature_path
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


def compute_aurc(probs, targets):
    """Computes Area Under the Risk-Coverage (AURC) directly on GPU."""
    confidences, preds = torch.max(probs, dim=1)
    errors = (preds != targets).float()
    
    # Sort by confidence descending
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_errors = errors[sorted_indices]
    
    # Calculate cumulative risk
    n_samples = len(targets)
    risks = torch.cumsum(sorted_errors, dim=0) / torch.arange(1, n_samples + 1, device=probs.device).float()
    
    # Area under curve calculation
    aurc = torch.sum(risks) / n_samples
    return aurc.item()


def compute_nll(probs, targets, num_classes=3):
    """Computes Negative Log-Likelihood (NLL)."""
    # probs shape: (N, num_classes)
    # targets shape: (N,)
    log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
    nll = -log_probs[torch.arange(len(targets)), targets].mean()
    return nll.item()


def compute_total_entropy(probs):
    """Computes Predictive Entropy (Total Entropy) per sample and mean."""
    # probs shape: (N, num_classes)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # (N,)
    return entropy


def compute_aleatoric_entropy(all_probs_list):
    """
    Computes Aleatoric Entropy (Expected Entropy) from MC dropout samples.
    
    Args:
        all_probs_list: List of (N, num_classes) tensors from MC passes
    
    Returns:
        Aleatoric entropy per sample, shape (N,)
    """
    # Stack to (num_passes, N, num_classes)
    stacked = torch.stack(all_probs_list, dim=0)
    
    # Compute entropy for each pass, then average
    # entropy shape per pass: (N,)
    entropies = -torch.sum(stacked * torch.log(stacked + 1e-10), dim=2)  # (num_passes, N)
    aleatoric = entropies.mean(dim=0)  # (N,)
    
    return aleatoric


def compute_mutual_information(all_probs_list, total_entropy):
    """
    Computes Mutual Information (Epistemic Uncertainty).
    MI = Total Entropy - Aleatoric Entropy
    
    Args:
        all_probs_list: List of (N, num_classes) tensors from MC passes
        total_entropy: Per-sample total entropy, shape (N,)
    
    Returns:
        Mutual information per sample, shape (N,)
    """
    aleatoric = compute_aleatoric_entropy(all_probs_list).to(total_entropy.device)
    epistemic = total_entropy - aleatoric
    return epistemic


def compute_uncertainty_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if INITIAL_CONFIG['mc_dropout'] and OOD_DATA:
        project_name = "mc_uncertainty_eval_ood"
    elif INITIAL_CONFIG['mc_dropout']:
        project_name = "mc_uncertainty_eval"
    elif OOD_DATA:
        project_name = "base_uncertainty_eval_ood"
    else:
        project_name = "base_uncertainty_eval"

    os.makedirs(SAVE_DIR_METRICS, exist_ok=True)
    
    fold_list = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("fold_")]
    fold_list.sort()
    
    # if INITIAL_CONFIG['mc_dropout']:
    #     temp_file_path = os.path.join(TEMPERATURES_PATH, "optimal_temperatures.json")
    #     with open(temp_file_path, "r") as f:
    #         optimal_temperatures = json.load(f)
    #     print(f"Loaded optimal temperatures from {temp_file_path}")

    # Determine targets once
    if OOD_DATA:
        target_names = sorted([p for p in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, p))])
    else:
        target_names = None

    # Dictionary to hold cross-fold metrics per target
    summary_metrics = {}

    for n, fold in enumerate(fold_list):
        print(f"Evaluating Fold {n} | MC Dropout: {INITIAL_CONFIG['mc_dropout']}")
        
        # 1. Determine exact test targets for this fold
        if OOD_DATA:
            current_targets = target_names
            current_dirs = [os.path.join(TEST_DATA_DIR, p) for p in current_targets]
            log_names = [f"{fold}_{p}" for p in current_targets]
        else:
            current_targets = ["id_test"]
            current_dirs = [os.path.join(TEST_DATA_DIR, fold)]
            log_names = [f"fold_{fold}"]

        checkpoint_fold_dir = os.path.join(CHECKPOINT_DIR, fold)
        checkpoint_path = os.path.join(checkpoint_fold_dir, os.listdir(checkpoint_fold_dir)[0])
        
        # Grab features shape dynamically
        features_shape = GraphDataset(current_dirs[0])[0].x.shape[-1]
        
        model = GATv2Lightning.load_from_checkpoint(
            checkpoint_path,
            in_features=features_shape,
            n_classes=3,
            n_gat_layers=INITIAL_CONFIG['n_gat_layers'],
            hidden_dim=INITIAL_CONFIG['hidden_dim'],
            n_heads=INITIAL_CONFIG['n_heads'],
            slope=INITIAL_CONFIG['slope'],
            dropout_on=INITIAL_CONFIG['mc_dropout'],
            pooling_method=INITIAL_CONFIG['pooling_method'],
            activation=INITIAL_CONFIG['activation'],
            norm_method=INITIAL_CONFIG['norm_method'],
            lr=INITIAL_CONFIG['lr'],
            weight_decay=INITIAL_CONFIG['weight_decay'],
            map_location=device,
        )
        
        # if INITIAL_CONFIG['mc_dropout']:
        #     model.temperature = optimal_temperatures[fold]

        wandb_logger = pl.loggers.WandbLogger(log_model=False)
        trainer = pl.Trainer(
            accelerator="auto", max_epochs=1, devices=1, enable_progress_bar=False,
            deterministic=False, logger=wandb_logger, enable_model_summary=False,
        )

        dir_fold = os.path.join(SAVE_DIR_METRICS, f"fold_{n}")
        os.makedirs(dir_fold, exist_ok=True)

        # Evaluate Each Target
        for t_name, t_dir, log_name in zip(current_targets, current_dirs, log_names):
            wandb.init(project=project_name, name=log_name, config=INITIAL_CONFIG)
            
            ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=15, norm='l1').to(device)

            dataset = GraphDataset(t_dir)
            loader = DataLoader(dataset, batch_size=1024, shuffle=False)

            if INITIAL_CONFIG['mc_dropout']:
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout') or 'GAT' in m.__class__.__name__:
                        m.train()
                        m.eval = types.MethodType(lambda self: self.train(), m)

            all_preds = []
            for p in range(50 if INITIAL_CONFIG['mc_dropout'] else 1):
                preds = trainer.predict(model, loader)
                preds = torch.cat(preds, dim=0)
                if INITIAL_CONFIG['mc_dropout']:
                    preds = torch.nn.functional.softmax(preds, dim=1)
                all_preds.append(preds)

            if INITIAL_CONFIG['mc_dropout']:
                preds_raw = torch.stack(all_preds).mean(dim=0)
            else:
                preds_raw = torch.nn.functional.softmax(all_preds[0], dim=1)

            preds_raw = preds_raw.to(device)
            ground_truth = torch.tensor([data.y.int().item() for data in dataset]).to(device)

            # =========================================================
            # COMPUTE UNCERTAINTY METRICS
            # =========================================================
            
            # ECE (Calibration)
            fold_ece = ece_metric(preds_raw, ground_truth).item()
            
            # NLL
            fold_nll = compute_nll(preds_raw, ground_truth, num_classes=3)
            
            # Total Entropy (Predictive Entropy)
            total_entropy_per_sample = compute_total_entropy(preds_raw)
            fold_total_entropy = total_entropy_per_sample.mean().item()
            
            # Aleatoric Entropy (Expected Entropy) - for MC Dropout
            if INITIAL_CONFIG['mc_dropout']:
                aleatoric_entropy_per_sample = compute_aleatoric_entropy(all_preds)
                fold_aleatoric_entropy = aleatoric_entropy_per_sample.mean().item()
                
                # Epistemic (Mutual Information)
                epistemic_entropy_per_sample = compute_mutual_information(all_preds, total_entropy_per_sample)
                fold_epistemic_entropy = epistemic_entropy_per_sample.mean().item()
            else:
                # For non-MC dropout, aleatoric = total, epistemic = 0
                aleatoric_entropy_per_sample = total_entropy_per_sample
                epistemic_entropy_per_sample = torch.zeros_like(total_entropy_per_sample)
                fold_aleatoric_entropy = fold_total_entropy
                fold_epistemic_entropy = 0.0
            
            # AURC
            fold_aurc = compute_aurc(preds_raw, ground_truth)
            
            # Log to W&B
            wandb.log({
                "ECE": fold_ece,
                "NLL": fold_nll,
                "Total_Entropy": fold_total_entropy,
                "Aleatoric_Entropy": fold_aleatoric_entropy,
                "Epistemic_Entropy": fold_epistemic_entropy,
                "AURC": fold_aurc,
            })
            wandb.finish()

            # Save fold-level scalar results
            fold_results = {
                "fold": fold,
                "target": t_name,
                "ECE": fold_ece,
                "NLL": fold_nll,
                "Total_Entropy": fold_total_entropy,
                "Aleatoric_Entropy": fold_aleatoric_entropy,
                "Epistemic_Entropy": fold_epistemic_entropy,
                "AURC": fold_aurc,
            }
            
            file_prefix = f"{t_name}_" if OOD_DATA else ""
            with open(os.path.join(dir_fold, f"{file_prefix}unc_results.json"), "w") as f:
                json.dump(fold_results, f)

            # Calucating predictions and correct/incorrect flag
            preds_class = preds_raw.argmax(dim=1)
            is_correct = (preds_class == ground_truth)

            # Save per-sample data for later plotting
            per_sample_data = {
                "sample_id": np.array([f"{fold}_{t_name}_{i}" for i in range(len(ground_truth))]), # Patient + Index
                "mean_probs": preds_raw.cpu().numpy(),
                "pred_labels": preds_class.cpu().numpy(),
                "true_labels": ground_truth.cpu().numpy(),
                "confidence": preds_raw.max(dim=1).values.cpu().numpy(),
                "is_correct": is_correct.cpu().numpy(),
                "total_entropy": total_entropy_per_sample.cpu().numpy(),
                "aleatoric_entropy": aleatoric_entropy_per_sample.cpu().numpy(),
                "epistemic_entropy": epistemic_entropy_per_sample.cpu().numpy(),
            }
            
            npz_path = os.path.join(dir_fold, f"{file_prefix}per_sample_data.npz")
            np.savez(npz_path, **per_sample_data)
            
            # Accumulate for cross-fold summary
            if t_name not in summary_metrics:
                summary_metrics[t_name] = {
                    "ece": [],
                    "nll": [],
                    "total_entropy": [],
                    "aleatoric_entropy": [],
                    "epistemic_entropy": [],
                    "aurc": [],
                }
            
            summary_metrics[t_name]["ece"].append(fold_ece)
            summary_metrics[t_name]["nll"].append(fold_nll)
            summary_metrics[t_name]["total_entropy"].append(fold_total_entropy)
            summary_metrics[t_name]["aleatoric_entropy"].append(fold_aleatoric_entropy)
            summary_metrics[t_name]["epistemic_entropy"].append(fold_epistemic_entropy)
            summary_metrics[t_name]["aurc"].append(fold_aurc)

    # =============================================================
    # FINAL SUMMARY LOGGING (per target if OOD, once if ID)
    # =============================================================
    for t_name, metrics in summary_metrics.items():
        log_name = f"summary_uncertainty_{t_name}" if OOD_DATA else "summary_uncertainty"
        wandb.init(project=project_name, name=log_name)
        
        summary_results = {
            "final_mean_ECE": mean(metrics["ece"]),
            "final_ECE_std": stdev(metrics["ece"]) if len(metrics["ece"]) > 1 else 0.0,
            
            "final_mean_NLL": mean(metrics["nll"]),
            "final_NLL_std": stdev(metrics["nll"]) if len(metrics["nll"]) > 1 else 0.0,
            
            "final_mean_Total_Entropy": mean(metrics["total_entropy"]),
            "final_Total_Entropy_std": stdev(metrics["total_entropy"]) if len(metrics["total_entropy"]) > 1 else 0.0,
            
            "final_mean_Aleatoric_Entropy": mean(metrics["aleatoric_entropy"]),
            "final_Aleatoric_Entropy_std": stdev(metrics["aleatoric_entropy"]) if len(metrics["aleatoric_entropy"]) > 1 else 0.0,
            
            "final_mean_Epistemic_Entropy": mean(metrics["epistemic_entropy"]),
            "final_Epistemic_Entropy_std": stdev(metrics["epistemic_entropy"]) if len(metrics["epistemic_entropy"]) > 1 else 0.0,
            
            "final_mean_AURC": mean(metrics["aurc"]),
            "final_AURC_std": stdev(metrics["aurc"]) if len(metrics["aurc"]) > 1 else 0.0,
        }
        
        wandb.log(summary_results)
        wandb.finish()
        
        file_suffix = f"_{t_name}" if OOD_DATA else ""
        with open(os.path.join(SAVE_DIR_METRICS, f"summary_unc_results{file_suffix}.json"), "w") as f:
            json.dump(summary_results, f)


if __name__ == "__main__":
    compute_uncertainty_metrics()
