import os
import torch
import numpy as np
from argparse import ArgumentParser
from src.utils.dataloader_utils import GraphDataset, save_data_list

class AddGaussianNoiseOOD:
    """
    Applies Signal-to-Noise Ratio (SNR) based Gaussian noise to EEG Graph Data.
    """
    def __init__(self, snr_db=0):
        self.snr_db = snr_db

    def __call__(self, data):
        # Clone to avoid modifying the cached data in GraphDataset memory
        new_data = data.clone()
        x = new_data.x
        
        # Calculate Signal Power
        signal_power = torch.mean(x**2)
        
        # Calculate Noise Power for target SNR
        # SNR_dB = 10 * log10(P_signal / P_noise)
        snr_linear = 10**(self.snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate and apply noise
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        new_data.x = x + noise
        
        return new_data

def generate_ood_from_folds(src_root, dest_root, snr_db):
    """
    Reads existing folds, applies noise to test sets, and saves OOD versions.
    """
    ood_transform = AddGaussianNoiseOOD(snr_db=snr_db)
    
    # Identify folds (fold_0, fold_1, etc.)
    folds = [f for f in os.listdir(src_root) if f.startswith("fold_")]
    
    for fold in sorted(folds):
        print(f"--- Processing {fold} ---")
        test_src_path = os.path.join(src_root, fold, "test")
        
        if not os.path.exists(test_src_path):
            print(f"Skipping {fold}: Test directory not found.")
            continue
            
        # 1. Load the test dataset using your existing GraphDataset class
        # We pass the transform so __getitem__ applies the noise
        test_dataset = GraphDataset(test_src_path)
        
        # 2. Generate the OOD samples
        print(f"Applying {snr_db}dB Gaussian Noise to {len(test_dataset)} samples...")
        ood_test_data = [ood_transform(test_dataset[i]) for i in range(len(test_dataset))]
        
        # 3. Define save path
        fold_dest_dir = os.path.join(dest_root, fold)
        os.makedirs(fold_dest_dir, exist_ok=True)
        save_path = os.path.join(fold_dest_dir, "test_data_ood.pt")
        
        # 4. Save using your existing utility
        save_data_list(ood_test_data, save_path)
        print(f"Saved OOD test set to: {save_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, 
                        help="Path to existing 'saved_folds' directory")
    parser.add_argument("--dest_dir", type=str, default="data/OOD_folds", 
                        help="Where to save the noisy OOD data")
    parser.add_argument("--snr", type=int, default=0, 
                        help="Signal-to-Noice Ratio in dB. Lower is noisier. 0 is a strong OOD signal.")
    
    args = parser.parse_args()
    
    generate_ood_from_folds(
        src_root=args.src_dir, 
        dest_root=args.dest_dir, 
        snr_db=args.snr
    )
    print("\nOOD Generation Complete.")
