import torch
from torch import optim
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl
import os
import json
from argparse import ArgumentParser

from src.models import GATv2Lightning
from src.utils.dataloader_utils import GraphDataset

def find_optimal_temperature(logits, labels, device):
    """Optimizes Temperature (T) using NLL on the validation set logits."""
    temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
    nll_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=200)

    def eval_fn():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    return temperature.item()

def calibrate_all_folds(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CHECKPOINT_DIR = args.checkpoint_dir
    FOLD_DATA_DIR = args.fold_data_dir
    SAVE_DIR = args.save_dir

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    fold_list = os.listdir(CHECKPOINT_DIR)
    checkpoint_fold_list = [os.path.join(CHECKPOINT_DIR, fold) for fold in fold_list]
    
    # Sort to ensure consistent fold_0 to fold_9 ordering
    fold_list.sort()
    checkpoint_fold_list.sort()
    
    optimal_temperatures = {}

    for n, fold in enumerate(fold_list):
        print(f"Calibrating Fold {n}...")
        
        # 1. Load the checkpoint
        checkpoint_path = os.path.join(
            checkpoint_fold_list[n], os.listdir(checkpoint_fold_list[n])[0]
        )
        
        # 2. Setup the Trainer
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=1,
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        
        # 3. Load the specific VALIDATION dataset for this fold
        val_data_path = os.path.join(FOLD_DATA_DIR, fold, "val")
        dataset = GraphDataset(val_data_path)
        features_shape = dataset[0].x.shape[-1]

        # 4. Load the Model with all architecture parameters!
        model = GATv2Lightning.load_from_checkpoint(
            checkpoint_path,
            in_features=features_shape,
            n_classes=3,
            n_gat_layers=1,
            hidden_dim=32,
            n_heads=9,
            slope=0.0025,
            dropout_on=args.dropout_on,  # Tied to the arg parser
            pooling_method="mean",
            activation="leaky_relu",
            norm_method="batch",
            lr=0.0012,
            weight_decay=0.0078,
            map_location=device,
        )
        
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)

        # 5. Get deterministic logits (MC Dropout is OFF by default during eval)
        preds = trainer.predict(model, loader)
        logits = torch.cat(preds, dim=0).to(device)
        
        labels = torch.tensor(
            [data.y.int().item() for data in dataset]
        ).to(device)

        # 6. Calculate Optimal Temperature
        optimal_T = find_optimal_temperature(logits, labels, device)
        print(f"Optimal Temperature for Fold {n}: {optimal_T:.4f}")
        
        # Save to dictionary
        optimal_temperatures[fold] = optimal_T

    # 7. Save the final JSON dictionary
    save_path = os.path.join(SAVE_DIR, "optimal_temperatures.json")
    with open(save_path, "w") as f:
        json.dump(optimal_temperatures, f, indent=4)
        
    print(f"\nCalibration complete! Saved to {save_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="saved_models/")
    parser.add_argument("--fold_data_dir", type=str, default="data/saved_folds/")
    parser.add_argument("--save_dir", type=str, default="save_temperature/")
    parser.add_argument("--dropout_on", action="store_true", default=False) # Add this!
    args = parser.parse_args()
    
    calibrate_all_folds(args)