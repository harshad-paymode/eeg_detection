import torch
from torch_geometric.loader import DataLoader
import numpy as np
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import json
import types
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import balanced_accuracy_score
from torchmetrics import Specificity, Recall, F1Score, AUROC
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
parser.add_argument("--save_dir_metrics", type=str, default="save_metrics/")
parser.add_argument("--mc_dropout", action="store_true", default=False)
parser.add_argument("--temperature_path", type=str, default="temperatures/")
parser.add_argument("--ood_data", action="store_true", default=False)
args = parser.parse_args()

CHECKPOINT_DIR = args.checkpoint_dir
TEST_DATA_DIR = args.test_data_dir
SAVE_DIR_METRICS = args.save_dir_metrics
TEMPERATURES_PATH = args.temperature_path
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
    
    summary_conf_matrix = np.zeros((3, 3))
    summary_balanced_acc, summary_specificity, summary_recall, summary_f1, summary_auroc = [], [], [], [], []

    if INITIAL_CONFIG['mc_dropout']:
        temp_file_path = os.path.join(TEMPERATURES_PATH, "optimal_temperatures.json")
        with open(temp_file_path, "r") as f:
            optimal_temperatures = json.load(f)

    for n, fold in enumerate(fold_list):
        print(f"Evaluating Fold {n} | MC Dropout: {INITIAL_CONFIG['mc_dropout']}")
        
        # 1. Determine exact test targets for this fold
        if OOD_DATA:
            # OOD targets: /ood_data/chb22, /ood_data/chb23, etc.
            patient_list = [p for p in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, p))]
            patient_list.sort()
            test_dirs = [os.path.join(TEST_DATA_DIR, p) for p in patient_list]
            log_names = [f"{fold}_{p}" for p in patient_list]
        else:
            # ID target: /test_data/fold_X
            test_dirs = [os.path.join(TEST_DATA_DIR, fold)]
            log_names = [f"fold_{fold}"]

        checkpoint_fold_dir = os.path.join(CHECKPOINT_DIR, fold)
        checkpoint_path = os.path.join(checkpoint_fold_dir, os.listdir(checkpoint_fold_dir)[0])
        
        # Grab features shape dynamically from the first target directory
        features_shape = GraphDataset(test_dirs[0])[0].x.shape[-1]
        
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
        
        if INITIAL_CONFIG['mc_dropout']:
            model.temperature = optimal_temperatures[fold]

        wandb_logger = pl.loggers.WandbLogger(log_model=False)
        trainer = pl.Trainer(
            accelerator="auto", max_epochs=1, devices=1, enable_progress_bar=False,
            deterministic=False, logger=wandb_logger, enable_model_summary=False,
        )

        fold_acc, fold_spec, fold_rec, fold_f1, fold_auc = [], [], [], [], []
        fold_conf_matrix = np.zeros((3, 3))

        # Evaluate Each Target (3 Patients for OOD, 1 Fold for ID)
        for t_dir, log_name in zip(test_dirs, log_names):
            wandb.init(project=project_name, name=log_name, config=INITIAL_CONFIG)
            
            conf_matrix_metric = MulticlassConfusionMatrix(3).to(device)
            specificity_metric = Specificity("multiclass", num_classes=3).to(device)
            recall_metric = Recall("multiclass", num_classes=3).to(device)
            f1_metric = F1Score("multiclass", num_classes=3).to(device)
            auroc_metric = AUROC("multiclass", num_classes=3).to(device)

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

            fold_acc.append(balanced_acc); fold_spec.append(specificity); fold_rec.append(recall)
            fold_f1.append(f1); fold_auc.append(auroc); fold_conf_matrix += conf_matrix

        # Aggregate & Save Average for the Fold
        dir_fold = os.path.join(SAVE_DIR_METRICS, f"fold_{n}")
        os.makedirs(dir_fold, exist_ok=True)
        
        fold_results = {
            "fold": fold,
            "AUROC": mean(fold_auc), "F1-score": mean(fold_f1),
            "Sensitivity": mean(fold_rec), "Specificity": mean(fold_spec),
            "Balanced Accuracy": mean(fold_acc)
        }
        
        np.save(os.path.join(dir_fold, "conf_matrix.npy"), fold_conf_matrix)
        with open(os.path.join(dir_fold, "results.json"), "w") as f:
            json.dump(fold_results, f)

        summary_conf_matrix += fold_conf_matrix
        summary_balanced_acc.append(mean(fold_acc))
        summary_specificity.append(mean(fold_spec))
        summary_recall.append(mean(fold_rec))
        summary_f1.append(mean(fold_f1))
        summary_auroc.append(mean(fold_auc))

    # Final Summary Logging
    wandb.init(project=project_name, name="summary_confusion_matrix")
    summary_results = {
        "final_mean_AUROC": mean(summary_auroc), "final_AUROC_std": stdev(summary_auroc),
        "final_mean_F1-score": mean(summary_f1), "final_F1-score_std": stdev(summary_f1),
        "final_mean_Sensitivity": mean(summary_recall), "final_Sensitivity_std": stdev(summary_recall),
        "final_Specificity": mean(summary_specificity), "final_Specificity_std": stdev(summary_specificity),
        "final_Balanced Accuracy": mean(summary_balanced_acc), "final_Balanced Accuracy_std": stdev(summary_balanced_acc),
    }
    wandb.log(summary_results)
    wandb.finish()
    
    np.save(os.path.join(SAVE_DIR_METRICS, "summary_conf_matrix.npy"), summary_conf_matrix)
    with open(os.path.join(SAVE_DIR_METRICS, "summary_results.json"), "w") as f:
        json.dump(summary_results, f)

if __name__ == "__main__":
    compute_prediction_metrics()

# import torch
# from torch_geometric.loader import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# from src.models import GATv2Lightning
# from src.utils.dataloader_utils import GraphDataset
# import lightning.pytorch as pl
# import os
# import json
# import types
# from torchmetrics.classification import MulticlassConfusionMatrix
# from sklearn.metrics import balanced_accuracy_score
# from torchmetrics import Accuracy, Specificity, Recall, F1Score, AUROC
# from argparse import ArgumentParser
# from statistics import mean, stdev
# import wandb

# api_key_file = open("/kaggle/working/eeg_detection/src/wandb_api_key.txt", "r")
# API_KEY = api_key_file.read()

# api_key_file.close()
# os.environ["WANDB_API_KEY"] = API_KEY


# parser = ArgumentParser()
# parser.add_argument( "--checkpoint_dir",type=str,default = "saved_models/")
# parser.add_argument("--test_data_dir",type=str,default = "test_data/")
# parser.add_argument("--save_dir_metrics",type=str,default="save_metrics/")
# parser.add_argument("--mc_dropout",action="store_true",default=False)
# parser.add_argument("--temperature_path",type=str,default ="temperatures/")
# parser.add_argument("--ood_data",action="store_true",default=False)
# args = parser.parse_args()

# CHECKPOINT_DIR = args.checkpoint_dir
# TEST_DATA_DIR = args.test_data_dir
# SAVE_DIR_METRICS = args.save_dir_metrics
# TEMPERATURES_PATH = args.temperature_path
# OOD_DATA = args.ood_data

# INITIAL_CONFIG = dict(
#     mc_dropout=args.mc_dropout,
#     n_gat_layers = 1,
#     hidden_dim = 32,
#     slope = 0.0025,
#     pooling_method = "mean",
#     norm_method = "batch",
#     activation = "leaky_relu",
#     n_heads = 9,
#     lr = 0.0012,
#     weight_decay = 0.0078
# )


# def compute_prediction_metrics():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     if INITIAL_CONFIG['mc_dropout'] and OOD_DATA:
#         project_name = "mc_confusoin_eval_ood"
#     elif INITIAL_CONFIG['mc_dropout']:
#         project_name = "mc_dropout_eval"
#     elif OOD_DATA:
#         project_name = "base_confusion_eval_ood"
#     else:
#         project_name = "base_confusion_eval"


#     os.makedirs(SAVE_DIR_METRICS,exist_ok = True)
    
#     fold_list = os.listdir(CHECKPOINT_DIR)
#     checkpoint_fold_list = [
#         os.path.join(CHECKPOINT_DIR, fold) for fold in fold_list
#     ]
#     data_fold_list = [os.path.join(TEST_DATA_DIR, fold) for fold in fold_list]
#     fold_list.sort()
#     data_fold_list.sort()
#     checkpoint_fold_list.sort()
    
#     summary_conf_matrix = np.zeros((3, 3))
#     summary_balanced_acc = []
#     summary_specificity = []
#     summary_recall = []
#     summary_f1 = []
#     summary_auroc = []
#     temp_file_path = ""

#     if INITIAL_CONFIG['mc_dropout']:
#         temp_file_path = os.path.join(TEMPERATURES_PATH, "optimal_temperatures.json")
#         with open(temp_file_path, "r") as f:
#             optimal_temperatures = json.load(f)
#         print(f"Loaded optimal temperatures from {temp_file_path}")
    
#     for n, fold in enumerate(fold_list):

#         wandb.init(
#             project=project_name,
#             name=f"fold_{fold}",
#             config=INITIAL_CONFIG,
#         )

#         CONFIG = wandb.config

#         conf_matrix_metric = MulticlassConfusionMatrix(3).to(device)
#         specificity_metric = Specificity("multiclass", num_classes=3).to(device)
#         recall_metric = Recall("multiclass", num_classes=3).to(device)
#         f1_metric = F1Score("multiclass", num_classes=3).to(device)
#         auroc_metric = AUROC("multiclass", num_classes=3).to(device)

#         print(f"Evaluating Fold {n} | MC Dropout: {CONFIG.mc_dropout}")
#         checkpoint_path = os.path.join(
#             checkpoint_fold_list[n], os.listdir(checkpoint_fold_list[n])[0]
#         )
#         wandb_logger = pl.loggers.WandbLogger(log_model=False)
#         trainer = pl.Trainer(
#             accelerator="auto",
#             max_epochs=1,
#             devices=1,
#             enable_progress_bar=False,
#             deterministic=False,
#             log_every_n_steps=50,
#             logger=wandb_logger,
#             enable_model_summary=False,
#         )
        
#         # Ensure GraphDataset only loads the specific test folder to prevent data leakage
#         dataset = GraphDataset(data_fold_list[n])
#         n_classes = 3
#         features_shape = dataset[0].x.shape[-1]

#         model = GATv2Lightning.load_from_checkpoint(
#             checkpoint_path,
#             in_features=features_shape,
#             n_classes=n_classes,
#             n_gat_layers=CONFIG.n_gat_layers,
#             hidden_dim=CONFIG.hidden_dim,
#             n_heads=CONFIG.n_heads,
#             slope=CONFIG.slope,
#             dropout_on=CONFIG.mc_dropout,
#             pooling_method=CONFIG.pooling_method,
#             activation=CONFIG.activation,
#             norm_method=CONFIG.norm_method,
#             lr=CONFIG.lr,
#             weight_decay=CONFIG.weight_decay,
#             map_location=device,
#         )
        
#         if CONFIG.mc_dropout:
#             model.temperature = optimal_temperatures[fold]
#             print(f"Applied Temperature Scaling: {model.temperature:.4f} for {fold}")

#         loader = DataLoader(dataset, batch_size=1024, shuffle=False)

#         # ---------------------------------------------------------
#         # THE MC DROPOUT EVALUATION ENGINE
#         # ---------------------------------------------------------
#         if CONFIG.mc_dropout:
#             # PyTorch Lightning to stop it from turning off Dropout during trainer.predict()
#             for m in model.modules():
#                 if m.__class__.__name__.startswith('Dropout'):
#                     m.train()
#                     m.eval = types.MethodType(lambda self: self.train(), m)

#         n_passes = 50 if CONFIG.mc_dropout else 1
#         all_preds = []

#         for p in range(n_passes):
#             preds = trainer.predict(model, loader)
#             preds = torch.cat(preds, dim=0)
            
#             if CONFIG.mc_dropout:
#                 # For MC Dropout, we must average probabilities, not raw logits
#                 preds = torch.nn.functional.softmax(preds, dim=1)
                
#             all_preds.append(preds)

#         if CONFIG.mc_dropout:
#             # Average the 50 probability distributions
#             preds_raw = torch.stack(all_preds).mean(dim=0)
#             preds = preds_raw.argmax(dim=1)
#         else:
#             # Standard single forward pass
#             preds_raw = all_preds[0]
#             preds = torch.nn.functional.softmax(preds_raw, dim=1).argmax(dim=1)
#         # ---------------------------------------------------------

#         # Add these two lines
#         preds_raw = preds_raw.to(device)
#         preds = preds.to(device)

#         ground_truth = torch.tensor(
#             [data.y.int().item() for data in dataset]
#         ).to(device)

#         conf_matrix = conf_matrix_metric(preds, ground_truth).cpu().int().numpy()
#         specificity = specificity_metric(preds, ground_truth).item()
#         recall = recall_metric(preds, ground_truth).item()
#         f1 = f1_metric(preds, ground_truth).item()
#         auroc = auroc_metric(preds_raw, ground_truth).item()
#         balanced_acc = balanced_accuracy_score(ground_truth.cpu(), preds.cpu())
        
#         summary_conf_matrix += conf_matrix
#         summary_balanced_acc.append(balanced_acc)
#         summary_specificity.append(specificity)
#         summary_recall.append(recall)
#         summary_f1.append(f1)
#         summary_auroc.append(auroc)

#         # saving fold results
#         fold_results = {
#             "fold": fold,
#             "AUROC": auroc,
#             "F1-score": f1,
#             "Sensitivity": recall,
#             "Specificity": specificity,
#             "Balanced Accuracy": balanced_acc,
#         }
        
#         wandb.log(fold_results)
#         wandb.finish()

#         dir_fold = os.path.join(SAVE_DIR_METRICS, f"fold_{n}")
#         os.makedirs(dir_fold,exist_ok=True)

#         np.save(
#             os.path.join(dir_fold, "conf_matrix.npy"),
#             conf_matrix,
#         )
#         with open(
#             os.path.join(dir_fold, f"results.json"), "w"
#         ) as f:
#             json.dump(fold_results, f)

#     # saving summary results
#     summary_results = {
#         "final_mean_AUROC": mean(summary_auroc),
#         "final_AUROC_std": stdev(summary_auroc),
#         "final_mean_F1-score": mean(summary_f1),
#         "final_F1-score_std": stdev(summary_f1),
#         "final_mean_Sensitivity": mean(summary_recall),
#         "final_Sensitivity_std": stdev(summary_recall),
#         "final_Specificity": mean(summary_specificity),
#         "final_Specificity_std": stdev(summary_specificity),
#         "final_Balanced Accuracy": mean(summary_balanced_acc),
#         "final_Balanced Accuracy_std": stdev(summary_balanced_acc),
#     }

#     # ---------------------------------------------------------
#     # FINAL SUMMARY LOGGING
#     # ---------------------------------------------------------
#     wandb.init(
#         project=project_name,
#         name=f"summary_unc_matrix"
#     )

#     wandb.log(summary_results)

#     wandb.finish()
    
#     np.save(
#         os.path.join(SAVE_DIR_METRICS, f"summary_conf_matrix.npy"),
#         summary_conf_matrix,
#     )

#     with open(
#         os.path.join(SAVE_DIR_METRICS, f"summary_results.json"), "w"
#     ) as f:
#         json.dump(summary_results, f)


# if __name__ == "__main__":
#     compute_prediction_metrics()
