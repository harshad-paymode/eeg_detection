import torch
from torch_geometric.loader import DataLoader
import numpy as np
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import torch_geometric
import json
import types
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import balanced_accuracy_score
from torchmetrics import Specificity, Recall, F1Score, AUROC
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
parser.add_argument("--save_dir_metrics", type=str, default="save_metrics/")
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

def compute_prediction_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if INITIAL_CONFIG['mc_dropout'] and OOD_DATA:
        project_name = "mc_confusion_eval_ood"
    elif INITIAL_CONFIG['mc_dropout']:
        project_name = "mc_dropout_eval"
    elif OOD_DATA:
        project_name = "base_confusion_eval_ood"
    else:
        project_name = "base_confusion_eval"

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

    # Dictionary to hold cross-fold metrics per target (patient or ID fold)
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

        # Evaluate Each Target (3 Patients for OOD, 1 Fold for ID)
        for t_name, t_dir, log_name in zip(current_targets, current_dirs, log_names):
            wandb.init(project=project_name, name=log_name, config=INITIAL_CONFIG)
            
            conf_matrix_metric = MulticlassConfusionMatrix(3).to(device)
            specificity_metric = Specificity("multiclass", num_classes=3).to(device)
            recall_metric = Recall("multiclass", num_classes=3).to(device)
            f1_metric = F1Score("multiclass", num_classes=3).to(device)
            auroc_metric = AUROC("multiclass", num_classes=3).to(device)

            dataset = GraphDataset(t_dir)
            loader = DataLoader(dataset, batch_size=1024, shuffle=False)

            # if INITIAL_CONFIG['mc_dropout']:
            #     model.train()
            #     for m in model.modules():
            #         if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch_geometric.nn.norm.BatchNorm):
            #             m.eval()

            # all_preds = []
            # for p in range(50 if INITIAL_CONFIG['mc_dropout'] else 1):
            #     preds = trainer.predict(model, loader)
            #     preds = torch.cat(preds, dim=0)
            #     if INITIAL_CONFIG['mc_dropout']:
            #         preds = torch.nn.functional.softmax(preds, dim=1)
            #     all_preds.append(preds)

            # if INITIAL_CONFIG['mc_dropout']:
            #     preds_raw = torch.stack(all_preds).mean(dim=0)
            #     preds = preds_raw.argmax(dim=1)
            # else:
            #     preds_raw = all_preds[0]
            #     preds = torch.nn.functional.softmax(preds_raw, dim=1).argmax(dim=1)


            # Set exactly the modes we want
            # if INITIAL_CONFIG['mc_dropout']:
            #     model.train()
            #     for m in model.modules():
            #         if isinstance(m, (torch.nn.BatchNorm1d, torch_geometric.nn.norm.BatchNorm)):
            #             m.eval()

            # else:
            #     model.eval()

            if INITIAL_CONFIG['mc_dropout']:
                model.train()
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout') or 'GAT' in m.__class__.__name__:
                        m.train()
                        m.eval = types.MethodType(lambda self: self.train(), m)

            all_preds = []
            
            # Manual inference loop (Bypasses Lightning completely)
            with torch.no_grad():
                for p in range(50 if INITIAL_CONFIG['mc_dropout'] else 1):
                    pass_preds = []
                    print("MODEL:", model.training)
                    for name, m in model.named_modules():
                        if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch_geometric.nn.norm.BatchNorm)) or "GATv2Conv" in type(m).__name__:
                            p = getattr(m, "p", None)
                            print(name, type(m).__name__, "training=", m.training, "p=", p)
                    for batch in loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        out1 = model(batch.x, batch.edge_index, batch.batch)
                        out2 = model(batch.x, batch.edge_index, batch.batch)
                        print("max diff:", (out1 - out2).abs().max().item())
                        pass_preds.append(out.detach().cpu())
                        
                    preds = torch.cat(pass_preds, dim=0)
                    
                    if INITIAL_CONFIG['mc_dropout']:
                        preds = torch.nn.functional.softmax(preds, dim=1)
                    
                    all_preds.append(preds)

            # 3. Aggregate
            if INITIAL_CONFIG['mc_dropout']:
                preds_raw = torch.stack(all_preds).mean(dim=0)
                preds = preds_raw.argmax(dim=1)
            else:
                preds_raw = all_preds[0]
                preds = torch.nn.functional.softmax(preds_raw, dim=1).argmax(dim=1)

            preds_raw, preds = preds_raw.to(device), preds.to(device)
            ground_truth = torch.tensor([data.y.int().item() for data in dataset]).to(device)

            conf_matrix = conf_matrix_metric(preds, ground_truth).cpu().int().numpy()
            specificity = specificity_metric(preds, ground_truth).item()
            recall = recall_metric(preds, ground_truth).item()
            f1 = f1_metric(preds, ground_truth).item()
            auroc = auroc_metric(preds_raw, ground_truth).item()
            balanced_acc = balanced_accuracy_score(ground_truth.cpu(), preds.cpu())

            wandb.log({
                "AUROC": auroc, "F1-score": f1, "Sensitivity": recall,
                "Specificity": specificity, "Balanced Accuracy": balanced_acc
            })
            wandb.finish()

            fold_results = {
                "fold": fold, "target": t_name,
                "AUROC": auroc, "F1-score": f1,
                "Sensitivity": recall, "Specificity": specificity,
                "Balanced Accuracy": balanced_acc
            }

            # Save fold-level results correctly named for OOD or ID
            file_prefix = f"{t_name}_" if OOD_DATA else ""
            np.save(os.path.join(dir_fold, f"{file_prefix}conf_matrix.npy"), conf_matrix)
            with open(os.path.join(dir_fold, f"{file_prefix}results.json"), "w") as f:
                json.dump(fold_results, f)

            # Accumulate for cross-fold summary
            if t_name not in summary_metrics:
                summary_metrics[t_name] = {
                    "auroc": [], "f1": [], "recall": [], 
                    "specificity": [], "bacc": [], "conf_matrix": np.zeros((3, 3))
                }
            
            summary_metrics[t_name]["auroc"].append(auroc)
            summary_metrics[t_name]["f1"].append(f1)
            summary_metrics[t_name]["recall"].append(recall)
            summary_metrics[t_name]["specificity"].append(specificity)
            summary_metrics[t_name]["bacc"].append(balanced_acc)
            summary_metrics[t_name]["conf_matrix"] += conf_matrix

    # Final Summary Logging (Iterating over patients if OOD, or just once if ID)
    for t_name, metrics in summary_metrics.items():
        log_name = f"summary_confusion_matrix_{t_name}" if OOD_DATA else "summary_confusion_matrix"
        wandb.init(project=project_name, name=log_name)
        
        summary_results = {
            "final_mean_AUROC": mean(metrics["auroc"]), "final_AUROC_std": stdev(metrics["auroc"]) if len(metrics["auroc"])>1 else 0.0,
            "final_mean_F1-score": mean(metrics["f1"]), "final_F1-score_std": stdev(metrics["f1"]) if len(metrics["f1"])>1 else 0.0,
            "final_mean_Sensitivity": mean(metrics["recall"]), "final_Sensitivity_std": stdev(metrics["recall"]) if len(metrics["recall"])>1 else 0.0,
            "final_Specificity": mean(metrics["specificity"]), "final_Specificity_std": stdev(metrics["specificity"]) if len(metrics["specificity"])>1 else 0.0,
            "final_Balanced Accuracy": mean(metrics["bacc"]), "final_Balanced Accuracy_std": stdev(metrics["bacc"]) if len(metrics["bacc"])>1 else 0.0,
        }
        wandb.log(summary_results)
        wandb.finish()
        
        file_suffix = f"_{t_name}" if OOD_DATA else ""
        np.save(os.path.join(SAVE_DIR_METRICS, f"summary_conf_matrix{file_suffix}.npy"), metrics["conf_matrix"])
        with open(os.path.join(SAVE_DIR_METRICS, f"summary_results{file_suffix}.json"), "w") as f:
            json.dump(summary_results, f)

if __name__ == "__main__":
    compute_prediction_metrics()