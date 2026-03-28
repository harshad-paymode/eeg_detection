import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import json
import types
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import balanced_accuracy_score
from torchmetrics import Accuracy, Specificity, Recall, F1Score, AUROC
from argparse import ArgumentParser
from statistics import mean, stdev
import wandb

api_key_file = open("/kaggle/working/eeg_detection/src/wandb_api_key.txt", "r")
API_KEY = api_key_file.read()

api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY


parser = ArgumentParser()
parser.add_argument( "--checkpoint_dir",type=str,default = "saved_models/")
parser.add_argument("--test_data_dir",type=str,default = "test_data/")
parser.add_argument("--save_dir_metrics",type=str,default="save_metrics/")
parser.add_argument("--mc_dropout",action="store_true",default=False)
args = parser.parse_args()

CHECKPOINT_DIR = args.checkpoint_dir
TEST_DATA_DIR = args.test_data_dir
SAVE_DIR_METRICS = args.save_dir_metrics

INITIAL_CONFIG = dict(
    mc_dropout=args.mc_dropout,
    n_gat_layers = 1,
    hidden_dim = 32,
    slope = 0.0025,
    pooling_method = "mean",
    norm_method = "batch",
    activation = "leaky_relu",
    n_heads = 9,
    lr = 0.0012,
    weight_decay = 0.0078
)


def compute_prediction_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name ="base_model_eval"

    os.makedirs(SAVE_DIR_METRICS,exist_ok = True)
    
    fold_list = os.listdir(CHECKPOINT_DIR)
    checkpoint_fold_list = [
        os.path.join(CHECKPOINT_DIR, fold) for fold in fold_list
    ]
    data_fold_list = [os.path.join(TEST_DATA_DIR, fold) for fold in fold_list]
    fold_list.sort()
    data_fold_list.sort()
    checkpoint_fold_list.sort()
    
    summary_conf_matrix = np.zeros((3, 3))
    summary_balanced_acc = []
    summary_specificity = []
    summary_recall = []
    summary_f1 = []
    summary_auroc = []

    if INITIAL_CONFIG['mc_dropout']:
        project_name = "mc_model_eval"

    wandb.init(
        project=project_name,
        name=f"confusion_matrix_and_accuracy",
        config= INITIAL_CONFIG,
    )

    CONFIG = wandb.config
    
    for n, fold in enumerate(fold_list):

        conf_matrix_metric = MulticlassConfusionMatrix(3).to(device)
        specificity_metric = Specificity("multiclass", num_classes=3).to(device)
        recall_metric = Recall("multiclass", num_classes=3).to(device)
        f1_metric = F1Score("multiclass", num_classes=3).to(device)
        auroc_metric = AUROC("multiclass", num_classes=3).to(device)

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
        
        # Ensure GraphDataset only loads the specific test folder to prevent data leakage
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
        
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)

        # ---------------------------------------------------------
        # THE MC DROPOUT EVALUATION ENGINE
        # ---------------------------------------------------------
        if CONFIG.mc_dropout:
            # PyTorch Lightning to stop it from turning off Dropout during trainer.predict()
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
                    m.eval = types.MethodType(lambda self: self.train(), m)

        n_passes = 50 if CONFIG.mc_dropout else 1
        all_preds = []

        for p in range(n_passes):
            preds = trainer.predict(model, loader)
            preds = torch.cat(preds, dim=0)
            
            if CONFIG.mc_dropout:
                # For MC Dropout, we must average probabilities, not raw logits
                preds = torch.nn.functional.softmax(preds, dim=1)
                
            all_preds.append(preds)

        if CONFIG.mc_dropout:
            # Average the 50 probability distributions
            preds_raw = torch.stack(all_preds).mean(dim=0)
            preds = preds_raw.argmax(dim=1)
        else:
            # Standard single forward pass
            preds_raw = all_preds[0]
            preds = torch.nn.functional.softmax(preds_raw, dim=1).argmax(dim=1)
        # ---------------------------------------------------------

        # Add these two lines
        preds_raw = preds_raw.to(device)
        preds = preds.to(device)

        ground_truth = torch.tensor(
            [data.y.int().item() for data in dataset]
        ).to(device)

        conf_matrix = conf_matrix_metric(preds, ground_truth).cpu().int().numpy()
        specificity = specificity_metric(preds, ground_truth).item()
        recall = recall_metric(preds, ground_truth).item()
        f1 = f1_metric(preds, ground_truth).item()
        auroc = auroc_metric(preds_raw, ground_truth).item()
        balanced_acc = balanced_accuracy_score(ground_truth.cpu(), preds.cpu())
        
        summary_conf_matrix += conf_matrix
        summary_balanced_acc.append(balanced_acc)
        summary_specificity.append(specificity)
        summary_recall.append(recall)
        summary_f1.append(f1)
        summary_auroc.append(auroc)

        # saving fold results
        fold_results = {
            "AUROC": auroc,
            "F1-score": f1,
            "Sensitivity": recall,
            "Specificity": specificity,
            "Balanced Accuracy": balanced_acc,
        }
        
        wandb.log(fold_results)


        dir_fold = os.path.join(SAVE_DIR_METRICS, f"fold_{n}")
        os.makedirs(dir_fold,exist_ok=True)

        np.save(
            os.path.join(dir_fold, "conf_matrix.npy"),
            conf_matrix,
        )
        with open(
            os.path.join(dir_fold, f"results.json"), "w"
        ) as f:
            json.dump(fold_results, f)

    # saving summary results
    summary_results = {
        "final_mean_AUROC": mean(summary_auroc),
        "final_AUROC_std": stdev(summary_auroc),
        "final_mean_F1-score": mean(summary_f1),
        "final_F1-score_std": stdev(summary_f1),
        "final_mean_Sensitivity": mean(summary_recall),
        "final_Sensitivity_std": stdev(summary_recall),
        "final_Specificity": mean(summary_specificity),
        "final_Specificity_std": stdev(summary_specificity),
        "final_Balanced Accuracy": mean(summary_balanced_acc),
        "final_Balanced Accuracy_std": stdev(summary_balanced_acc),
    }

    wandb.log(summary_results)

    wandb.finish()
    
    np.save(
        os.path.join(SAVE_DIR_METRICS, f"summary_conf_matrix.npy"),
        summary_conf_matrix,
    )

    with open(
        os.path.join(SAVE_DIR_METRICS, f"summary_results.json"), "w"
    ) as f:
        json.dump(summary_results, f)


if __name__ == "__main__":
    compute_prediction_metrics()