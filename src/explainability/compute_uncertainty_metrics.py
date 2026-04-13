import torch
from torch_geometric.loader import DataLoader
import numpy as np
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import json
import types
import torch_geometric
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
    
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_errors = errors[sorted_indices]
    
    n_samples = len(targets)
    risks = torch.cumsum(sorted_errors, dim=0) / torch.arange(1, n_samples + 1, device=probs.device).float()
    
    aurc = torch.sum(risks) / n_samples
    return aurc.item()

def compute_nll(probs, targets, num_classes=3):
    """Computes Negative Log-Likelihood (NLL). Clamped to prevent NaN."""
    probs = torch.clamp(probs, min=1e-7, max=1.0)
    log_probs = torch.log(probs) 
    nll_vals = -log_probs[torch.arange(len(targets)), targets]
    return torch.nanmean(nll_vals).item()

def compute_total_entropy(probs):
    """Computes Predictive Entropy. Clamped to prevent NaN."""
    probs = torch.clamp(probs, min=1e-7, max=1.0)
    entropy = -torch.sum(probs * torch.log(probs), dim=1) 
    return entropy

def compute_aleatoric_entropy(all_probs_list):
    """Computes Aleatoric Entropy from MC dropout stochastic passes."""
    stacked = torch.stack(all_probs_list, dim=0)
    stacked = torch.clamp(stacked, min=1e-7, max=1.0)
    entropies = -torch.sum(stacked * torch.log(stacked), dim=2) 
    aleatoric = torch.nanmean(entropies, dim=0) 
    return aleatoric

def compute_mutual_information(all_probs_list, total_entropy):
    """Computes Epistemic Uncertainty (Mutual Information)."""
    aleatoric = compute_aleatoric_entropy(all_probs_list).to(total_entropy.device)
    epistemic = total_entropy - aleatoric
    epistemic = torch.clamp(epistemic, min=0.0)
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

    if OOD_DATA:
        target_names = sorted([p for p in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, p))])
    else:
        target_names = None

    # Tracking variable for Ensemble
    ensemble_tracking = {}
    if OOD_DATA:
        for t in target_names:
            ensemble_tracking[t] = {"all_passes": [], "ground_truth": None}

    summary_metrics = {}

    for n, fold in enumerate(fold_list):
        print(f"\n{'='*50}")
        print(f"Evaluating Fold {n} | MC Dropout: {INITIAL_CONFIG['mc_dropout']}")
        
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

        for t_name, t_dir, log_name in zip(current_targets, current_dirs, log_names):
            wandb.init(project=project_name, name=log_name, config=INITIAL_CONFIG)
            
            ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=15, norm='l1').to(device)

            dataset = GraphDataset(t_dir)
            loader = DataLoader(dataset, batch_size=1024, shuffle=False)

            if INITIAL_CONFIG['mc_dropout']:
                model.train()
                for m in model.modules():
                    if isinstance(m, (torch.nn.BatchNorm1d, torch_geometric.nn.norm.BatchNorm)):
                        m.eval()
            else:
                model.eval()

            all_preds = []
            
            with torch.no_grad():
                for p in range(50 if INITIAL_CONFIG['mc_dropout'] else 1):
                    pass_preds = []
                    for batch in loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        pass_preds.append(out.detach().cpu())
                        
                    preds = torch.cat(pass_preds, dim=0)
                    if INITIAL_CONFIG['mc_dropout']:
                        preds = torch.nn.functional.softmax(preds, dim=1)
                    all_preds.append(preds)

            if INITIAL_CONFIG['mc_dropout']:
                preds_raw = torch.nanmean(torch.stack(all_preds), dim=0)
                preds_class = preds_raw.argmax(dim=1)
            else:
                preds_raw = torch.nn.functional.softmax(all_preds[0], dim=1) 
                preds_class = preds_raw.argmax(dim=1)

            preds_raw, preds_class = preds_raw.to(device), preds_class.to(device)
            ground_truth = torch.tensor([data.y.int().item() for data in dataset]).to(device)

            if OOD_DATA:
                if INITIAL_CONFIG['mc_dropout']:
                    ensemble_tracking[t_name]["all_passes"].extend([p.detach().cpu() for p in all_preds])
                else:
                    ensemble_tracking[t_name]["all_passes"].append(preds_raw.detach().cpu())
                    
                if ensemble_tracking[t_name]["ground_truth"] is None:
                    ensemble_tracking[t_name]["ground_truth"] = ground_truth.detach().cpu()

            # COMPUTE UNCERTAINTY METRICS
            fold_ece = ece_metric(preds_raw, ground_truth).item()
            fold_nll = compute_nll(preds_raw, ground_truth, num_classes=3)
            
            total_entropy_per_sample = compute_total_entropy(preds_raw)
            fold_total_entropy = torch.nanmean(total_entropy_per_sample).item()
            
            if INITIAL_CONFIG['mc_dropout']:
                aleatoric_entropy_per_sample = compute_aleatoric_entropy(all_preds)
                fold_aleatoric_entropy = torch.nanmean(aleatoric_entropy_per_sample).item()
                
                epistemic_entropy_per_sample = compute_mutual_information(all_preds, total_entropy_per_sample)
                fold_epistemic_entropy = torch.nanmean(epistemic_entropy_per_sample).item()
            else:
                aleatoric_entropy_per_sample = total_entropy_per_sample
                epistemic_entropy_per_sample = torch.zeros_like(total_entropy_per_sample)
                fold_aleatoric_entropy = fold_total_entropy
                fold_epistemic_entropy = 0.0
            
            fold_aurc = compute_aurc(preds_raw, ground_truth)
            
            # Explicit Console Print for user visibility
            print(f"  -> Target: {t_name} | ECE: {fold_ece:.4f} | NLL: {fold_nll:.4f} | Total Ent: {fold_total_entropy:.4f}")

            wandb.log({
                "ECE": fold_ece, "NLL": fold_nll, "Total_Entropy": fold_total_entropy,
                "Aleatoric_Entropy": fold_aleatoric_entropy, "Epistemic_Entropy": fold_epistemic_entropy,
                "AURC": fold_aurc,
            })
            wandb.finish()

            fold_results = {
                "fold": fold, "target": t_name, "ECE": fold_ece, "NLL": fold_nll,
                "Total_Entropy": fold_total_entropy, "Aleatoric_Entropy": fold_aleatoric_entropy,
                "Epistemic_Entropy": fold_epistemic_entropy, "AURC": fold_aurc,
            }
            
            file_prefix = f"{t_name}_" if OOD_DATA else ""
            with open(os.path.join(dir_fold, f"{file_prefix}unc_results.json"), "w") as f:
                json.dump(fold_results, f)

            is_correct = (preds_class == ground_truth)

            per_sample_data = {
                "sample_id": np.array([f"{fold}_{t_name}_{i}" for i in range(len(ground_truth))]), 
                "mean_probs": preds_raw.cpu().numpy(),
                "pred_labels": preds_class.cpu().numpy(),
                "true_labels": ground_truth.cpu().numpy(),
                "confidence": preds_raw.max(dim=1).values.cpu().numpy(),
                "is_correct": is_correct.cpu().numpy(),
                "total_entropy": total_entropy_per_sample.cpu().numpy(),
                "aleatoric_entropy": aleatoric_entropy_per_sample.cpu().numpy(),
                "epistemic_entropy": epistemic_entropy_per_sample.cpu().numpy(),
            }
            
            np.savez(os.path.join(dir_fold, f"{file_prefix}per_sample_data.npz"), **per_sample_data)
            
            if t_name not in summary_metrics:
                summary_metrics[t_name] = {"ece": [], "nll": [], "total_entropy": [], "aleatoric_entropy": [], "epistemic_entropy": [], "aurc": []}
            
            summary_metrics[t_name]["ece"].append(fold_ece)
            summary_metrics[t_name]["nll"].append(fold_nll)
            summary_metrics[t_name]["total_entropy"].append(fold_total_entropy)
            summary_metrics[t_name]["aleatoric_entropy"].append(fold_aleatoric_entropy)
            summary_metrics[t_name]["epistemic_entropy"].append(fold_epistemic_entropy)
            summary_metrics[t_name]["aurc"].append(fold_aurc)

    # =============================================================
    # FINAL SUMMARY LOGGING 
    # =============================================================
    for t_name, metrics in summary_metrics.items():
        log_name = f"summary_uncertainty_{t_name}" if OOD_DATA else "summary_uncertainty"
        wandb.init(project=project_name, name=log_name)
        
        summary_results = {
            "final_mean_ECE": mean(metrics["ece"]), "final_ECE_std": stdev(metrics["ece"]) if len(metrics["ece"]) > 1 else 0.0,
            "final_mean_NLL": mean(metrics["nll"]), "final_NLL_std": stdev(metrics["nll"]) if len(metrics["nll"]) > 1 else 0.0,
            "final_mean_Total_Entropy": mean(metrics["total_entropy"]), "final_Total_Entropy_std": stdev(metrics["total_entropy"]) if len(metrics["total_entropy"]) > 1 else 0.0,
            "final_mean_Aleatoric_Entropy": mean(metrics["aleatoric_entropy"]), "final_Aleatoric_Entropy_std": stdev(metrics["aleatoric_entropy"]) if len(metrics["aleatoric_entropy"]) > 1 else 0.0,
            "final_mean_Epistemic_Entropy": mean(metrics["epistemic_entropy"]), "final_Epistemic_Entropy_std": stdev(metrics["epistemic_entropy"]) if len(metrics["epistemic_entropy"]) > 1 else 0.0,
            "final_mean_AURC": mean(metrics["aurc"]), "final_AURC_std": stdev(metrics["aurc"]) if len(metrics["aurc"]) > 1 else 0.0,
        }
        
        wandb.log(summary_results)
        wandb.finish()
        
        file_suffix = f"_{t_name}" if OOD_DATA else ""
        with open(os.path.join(SAVE_DIR_METRICS, f"summary_unc_results{file_suffix}.json"), "w") as f:
            json.dump(summary_results, f)

    # =========================================================================
    # ENSEMBLE CALCULATIONS (OOD DATA ONLY)
    # =========================================================================
    if OOD_DATA:
        print("\n" + "="*70)
        print("FINAL ENSEMBLE UNCERTAINTY RESULTS (Pooled Passes Across Folds)")
        print("="*70)
        
        for t_name, data in ensemble_tracking.items():
            if not data["all_passes"]:
                continue
                
            all_passes = data["all_passes"]
            gt = data["ground_truth"]
            
            stacked_probs = torch.stack(all_passes, dim=0) 
            ensemble_mean_probs = torch.nanmean(stacked_probs, dim=0) # Safe average across all pooled folds
            ensemble_preds_class = ensemble_mean_probs.argmax(dim=1)
            
            ece_metric_ens = MulticlassCalibrationError(num_classes=3, n_bins=15, norm='l1')
            
            ens_ece = ece_metric_ens(ensemble_mean_probs, gt).item()
            ens_nll = compute_nll(ensemble_mean_probs, gt, num_classes=3)
            ens_aurc = compute_aurc(ensemble_mean_probs, gt)
            
            total_entropy_per_sample = compute_total_entropy(ensemble_mean_probs)
            ens_total_entropy = torch.nanmean(total_entropy_per_sample).item()
            
            if INITIAL_CONFIG['mc_dropout']:
                aleatoric_entropy_per_sample = compute_aleatoric_entropy(all_passes)
                ens_aleatoric_entropy = torch.nanmean(aleatoric_entropy_per_sample).item()
                
                epistemic_entropy_per_sample = compute_mutual_information(all_passes, total_entropy_per_sample)
                ens_epistemic_entropy = torch.nanmean(epistemic_entropy_per_sample).item()
            else:
                aleatoric_entropy_per_sample = total_entropy_per_sample
                epistemic_entropy_per_sample = torch.zeros_like(total_entropy_per_sample)
                ens_aleatoric_entropy = ens_total_entropy
                ens_epistemic_entropy = 0.0
                
            print(f"\n--- OOD Patient: {t_name} (Total Pooled Passes: {len(all_passes)}) ---")
            print(f"ECE               : {ens_ece:.4f}")
            print(f"NLL               : {ens_nll:.4f}")
            print(f"AURC              : {ens_aurc:.4f}")
            print(f"Total Entropy     : {ens_total_entropy:.4f}")
            print(f"Aleatoric Entropy : {ens_aleatoric_entropy:.4f}")
            print(f"Epistemic Entropy : {ens_epistemic_entropy:.4f}")
            
            ens_results = {
                "ECE": ens_ece, "NLL": ens_nll, "AURC": ens_aurc,
                "Total_Entropy": ens_total_entropy, "Aleatoric_Entropy": ens_aleatoric_entropy,
                "Epistemic_Entropy": ens_epistemic_entropy
            }
            
            with open(os.path.join(SAVE_DIR_METRICS, f"ensemble_unc_results_{t_name}.json"), "w") as f:
                json.dump(ens_results, f, indent=4)
                
            confidences = ensemble_mean_probs.max(dim=1).values.numpy()
            is_correct = (ensemble_preds_class == gt).numpy()
            
            ens_per_sample_data = {
                "sample_id": np.array([f"ens_{t_name}_{i}" for i in range(len(gt))]),
                "mean_probs": ensemble_mean_probs.numpy(),
                "pred_labels": ensemble_preds_class.numpy(),
                "true_labels": gt.numpy(),
                "confidence": confidences,
                "is_correct": is_correct,
                "total_entropy": total_entropy_per_sample.numpy(),
                "aleatoric_entropy": aleatoric_entropy_per_sample.numpy(),
                "epistemic_entropy": epistemic_entropy_per_sample.numpy(),
            }
            
            np.savez(os.path.join(SAVE_DIR_METRICS, f"ensemble_per_sample_data_{t_name}.npz"), **ens_per_sample_data)
            
        print("="*70 + "\n")

if __name__ == "__main__":
    compute_uncertainty_metrics()