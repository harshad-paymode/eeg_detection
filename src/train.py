import os
import warnings
import multiprocessing as mp
from argparse import ArgumentParser
from statistics import mean, stdev
import torch
import torch_geometric
import wandb
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from lightning.pytorch.callbacks import TQDMProgressBar

from models import GATv2Lightning


warnings.filterwarnings(
    "ignore", ".*does not have many workers.*"
)  # DISABLED ON PURPOSE
torch_geometric.seed_everything(42)
api_key_file = open("/kaggle/working/eeg_detection/src/wandb_api_key.txt", "r")
API_KEY = api_key_file.read()

api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("medium")
parser = ArgumentParser()
 
parser.add_argument("--fft", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--saved_models_dir",type=str,default="data/models")
parser.add_argument("--fold_data_dir",type=str,default = "data/saved_folds")
parser.add_argument("--train_test_split", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--weights", action="store_true", default=False)
parser.add_argument("--use_ictal_periods", action="store_true", default=False)
parser.add_argument("--use_preictal_periods", action="store_true", default=False)
parser.add_argument("--use_interictal_periods", action="store_true", default=False)
parser.add_argument("--dropout_on",action="store_true",default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_splits", type=int, default=10)

args = parser.parse_args()

FOLD_DATA_DIR = args.fold_data_dir
SAVE_MODELS_PATH = args.saved_models_dir
WEIGHTS_FLAG = args.weights
EPOCHS = args.epochs
FFT = args.fft
USED_CLASSES_DICT = {
    "ictal": args.use_ictal_periods,
    "interictal": args.use_interictal_periods,
    "preictal": args.use_preictal_periods,
}
TRAIN_VAL_SPLIT = args.train_test_split
BATCH_SIZE = args.batch_size

SEED = args.seed
KFOLD_CVAL_MODE = True
N_SPLITS = args.n_splits
INITIAL_CONFIG = dict(
    weights=WEIGHTS_FLAG,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    fft=FFT,
    used_classes_dict=USED_CLASSES_DICT,
    train_val_split=TRAIN_VAL_SPLIT,
    seed=SEED,
    n_gat_layers=1,
    hidden_dim=32,
    dropout_on=args.dropout_on,
    slope=0.0025,
    pooling_method="mean",
    norm_method="batch",
    activation="leaky_relu",
    n_heads=9,
    lr=0.0012,
    weight_decay=0.0078,
)



def train_kfold_cval():
    """Load saved splits and initialize kfold cross validation training."""
    torch_geometric.seed_everything(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    precision = "bf16-mixed" if device == "cpu" else "16-mixed"
    strategy = pl.strategies.SingleDeviceStrategy(device=torch_device)
    
    result_list_auroc = []
    result_list_f1 = []
    
    for fold in range(N_SPLITS):
        print(f"Training Fold {fold}")
        if INITIAL_CONFIG.dropout_on:
            wandb.init(
                project="eeg_dropout_model",
                name=f"fold_{fold}",
                config=INITIAL_CONFIG,
            )
        else:
            wandb.init(
                project="eeg_base_model",
                name=f"fold_{fold}",
                config=INITIAL_CONFIG,
            )
        CONFIG = wandb.config
        
        # Load exactly the same data for all experiments
        fold_dir = os.path.join(FOLD_DATA_DIR, f"fold_{fold}")
        train_dataset = torch.load(os.path.join(fold_dir, "train_data.pt"))
        valid_dataset = torch.load(os.path.join(fold_dir, "valid_data.pt"))
        test_data = torch.load(os.path.join(fold_dir, "test_data.pt"))

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )
        test_dataloader = DataLoader(
            test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        train_labels = torch.cat([data.y for data in train_dataset])
        label_properties = torch.unique(train_labels, return_counts=True)
        
        if sum(CONFIG.used_classes_dict.values()) == 3:
            """Multiclass weights"""
            class_weight = torch.from_numpy(
                compute_class_weight(
                    "balanced",
                    classes=label_properties[0].numpy(),
                    y=train_labels.numpy(),
                )
            ).float()
        else:
            """Binary weights"""
            class_weight = torch.tensor(
                [label_properties[1][0] / label_properties[1][1]]
            )
            
        n_classes = sum(USED_CLASSES_DICT.values())
        features_shape = train_dataset[0].x.shape[-1]
        
        wandb_logger = pl.loggers.WandbLogger(log_model=False)
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=False, mode="min"
        )
        best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            verbose=False,
            dirpath=f"{SAVE_MODELS_PATH}/fold_{fold}",
            filename="{epoch}-{val_loss:.3f}",
        )
        callbacks = [early_stopping, best_checkpoint_callback]
        
        trainer = pl.Trainer(
            accelerator="auto",
            precision=precision,
            devices=1,
            max_epochs=EPOCHS,
            enable_progress_bar=False,
            strategy=strategy,
            deterministic=False,
            log_every_n_steps=50,
            enable_model_summary=True,
            logger=wandb_logger,
            callbacks=callbacks,
        )

        model = GATv2Lightning(
            features_shape,
            n_classes=n_classes,
            n_gat_layers=CONFIG.n_gat_layers,
            hidden_dim=CONFIG.hidden_dim,
            n_heads=CONFIG.n_heads,
            slope=CONFIG.slope,
            dropout_on=CONFIG.dropout_on,
            pooling_method=CONFIG.pooling_method,
            activation=CONFIG.activation,
            norm_method=CONFIG.norm_method,
            lr=CONFIG.lr,
            weight_decay=CONFIG.weight_decay,
            fft_mode=FFT,
            class_weights=class_weight,
        )
        
        trainer.fit(model, train_dataloader, valid_dataloader)
        eval_results = trainer.test(model, test_dataloader, ckpt_path="best")[0]

        fold_auroc = eval_results.get("test_AUROC", 0)
        result_list_auroc.append(fold_auroc)

        fold_f1 = eval_results.get("test_f1_score", 0)
        result_list_f1.append(fold_f1)
        
        if fold == N_SPLITS - 1:
            # logging final auroc
            mean_auroc = mean(result_list_auroc)
            stdev_auroc = round(stdev(result_list_auroc), 4)
            sem_auroc = round(stdev_auroc / (len(result_list_auroc) ** 0.5), 4) if len(result_list_auroc) > 0 else 0
            wandb.log({
                "final_mean_AUROC": mean_auroc,
                "final_stdev_AUROC": stdev_auroc,
                "final_sem_AUROC": sem_auroc
            }) 

            # logging final f1 score
            mean_f1 = mean(result_list_f1)
            stdev_f1 = round(stdev(result_list_f1), 4)
            sem_f1 = round(stdev_f1 / (len(result_list_f1) ** 0.5), 4) if len(result_list_f1) > 0 else 0
            wandb.log({
                "final_mean_f1": mean_f1,
                "final_stdev_f1": stdev_f1,
                "final_sem_f1": sem_f1
            })        

        wandb.finish()

    return None


if __name__ == "__main__":
    train_kfold_cval()