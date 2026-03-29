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
from torchmetrics.regression import MeanSquaredError
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
parser.add_argument("--temperature_path",type=str,default ="temperatures/")
args = parser.parse_args()

CHECKPOINT_DIR = args.checkpoint_dir
TEST_DATA_DIR = args.test_data_dir
SAVE_DIR_METRICS = args.save_dir_metrics
TEMPERATURES_PATH = args.temperature_path

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

def compute_uncertainty_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = "mc_uncertainty_eval_ood" if INITIAL_CONFIG['mc_dropout'] else "base_uncertainty_eval_ood"

    os.makedirs(SAVE_DIR_METRICS, exist_ok=True)
    
    fold_list = os.listdir(CHECKPOINT_DIR)
    checkpoint_fold_list = [os.path.join(CHECKPOINT_DIR, fold) for fold in fold_list]
    data_fold_list = [os.path.join(TEST_DATA_DIR, fold) for fold in fold_list]
    
    fold_list.sort()
    data_fold_list.sort()
    checkpoint_fold_list.sort()
    
    summary_ece = []
    summary_brier = []
    summary_aurc = []
    temp_file_path = ""
    
    if INITIAL_CONFIG['mc_dropout']:
        temp_file_path = os.path.join(TEMPERATURES_PATH, "optimal_temperatures.json")
        with open(temp_file_path, "r") as f:
            optimal_temperatures = json.load(f)
        print(f"Loaded optimal temperatures from {temp_file_path}")
    
    for n, fold in enumerate(fold_list):

        ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=15, norm='l1').to(device)
        brier_metric = MeanSquaredError().to(device)

        # Init W&B for this specific fold
        wandb.init(
            project=project_name,
            name=f"fold_{fold}",
            config=INITIAL_CONFIG,
        )

        CONFIG = wandb.config
        print(f"Evaluating Fold {n} | MC Dropout: {CONFIG.mc_dropout}")
        checkpoint_path = os.path.join(
            checkpoint_fold_list[n], os.listdir(checkpoint_fold_list[n])[0]
        )
        
        wandb_logger = pl.loggers.WandbLogger(log_model=False)
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=1,
            devices=1,
            enable_progress_bar=False,
            deterministic=False,
            log_every_n_steps=50,
            logger=wandb_logger,
            enable_model_summary=False,
        )
        
        dataset = GraphDataset(data_fold_list[n])
        n_classes = 3
        features_shape = dataset[0].x.shape[-1]

        model = GATv2Lightning.load_from_checkpoint(
            checkpoint_path,
            in_features=features_shape,
            n_classes=n_classes,
            n_gat_layers=CONFIG.n_gat_layers,
            hidden_dim=CONFIG.hidden_dim,
            n_heads=CONFIG.n_heads,
            slope=CONFIG.slope,
            dropout_on=CONFIG.mc_dropout,
            pooling_method=CONFIG.pooling_method,
            activation=CONFIG.activation,
            norm_method=CONFIG.norm_method,
            lr=CONFIG.lr,
            weight_decay=CONFIG.weight_decay,
            map_location=device,
        )
        
        if CONFIG.mc_dropout:
            model.temperature = optimal_temperatures[fold]
            print(f"Applied Temperature Scaling: {model.temperature:.4f} for {fold}")

        loader = DataLoader(dataset, batch_size=1024, shuffle=False)

        # ---------------------------------------------------------
        # THE MC DROPOUT EVALUATION ENGINE
        # ---------------------------------------------------------
        if CONFIG.mc_dropout:
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout') or 'GAT' in m.__class__.__name__:
                    m.train()
                    m.eval = types.MethodType(lambda self: self.train(), m)

        n_passes = 50 if CONFIG.mc_dropout else 1
        all_preds = []

        for p in range(n_passes):
            preds = trainer.predict(model, loader)
            preds = torch.cat(preds, dim=0)
            
            if CONFIG.mc_dropout:
                preds = torch.nn.functional.softmax(preds, dim=1)
                
            all_preds.append(preds)

        if CONFIG.mc_dropout:
            preds_raw = torch.stack(all_preds).mean(dim=0)
        else:
            preds_raw = torch.nn.functional.softmax(all_preds[0], dim=1)
        # ---------------------------------------------------------

        preds_raw = preds_raw.to(device)
        ground_truth = torch.tensor(
            [data.y.int().item() for data in dataset]
        ).to(device)

        # Calculate Uncertainty Metrics
        fold_ece = ece_metric(preds_raw, ground_truth).item()
        ground_truth_one_hot = torch.nn.functional.one_hot(ground_truth.long(), num_classes=3).float()
        fold_brier = brier_metric(preds_raw, ground_truth_one_hot).item()
        fold_aurc = compute_aurc(preds_raw, ground_truth)
        
        summary_ece.append(fold_ece)
        summary_brier.append(fold_brier)
        summary_aurc.append(fold_aurc)

        # saving fold results
        fold_results = {
            "fold": fold,
            "ECE": fold_ece,
            "Brier_Score": fold_brier,
            "AURC": fold_aurc,
        }
        
        wandb.log(fold_results)
        wandb.finish() # Closes the fold run perfectly

        dir_fold = os.path.join(SAVE_DIR_METRICS, f"fold_{n}")
        os.makedirs(dir_fold, exist_ok=True)

        with open(os.path.join(dir_fold, f"unc_results.json"), "w") as f:
            json.dump(fold_results, f)

    # ---------------------------------------------------------
    # FINAL SUMMARY LOGGING
    # ---------------------------------------------------------
    wandb.init(
        project=project_name,
        name=f"summary_unc_matrix"
    )
    
    summary_results = {
        "ECE": mean(summary_ece),
        "ECE_std": stdev(summary_ece),
        "Brier_Score": mean(summary_brier),
        "Brier_Score_std": stdev(summary_brier),
        "AURC": mean(summary_aurc),
        "AURC_std": stdev(summary_aurc),
    }

    wandb.log(summary_results)
    wandb.finish()

    with open(os.path.join(SAVE_DIR_METRICS, f"summary_unc_results.json"), "w") as f:
        json.dump(summary_results, f)


if __name__ == "__main__":
    compute_uncertainty_metrics()