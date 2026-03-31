# import os
# import torch
# import numpy as np
# from argparse import ArgumentParser
# from src.utils.dataloader_utils import GraphDataset, save_data_list

# class AddGaussianNoiseOOD:
#     """
#     Applies Signal-to-Noise Ratio (SNR) based Gaussian noise to EEG Graph Data.
#     """
#     def __init__(self, snr_db=0):
#         self.snr_db = snr_db

#     def __call__(self, data):
#         # Clone to avoid modifying the cached data in GraphDataset memory
#         new_data = data.clone()
#         x = new_data.x
        
#         # Calculate Signal Power
#         signal_power = torch.mean(x**2)
        
#         # Calculate Noise Power for target SNR
#         # SNR_dB = 10 * log10(P_signal / P_noise)
#         snr_linear = 10**(self.snr_db / 10.0)
#         noise_power = signal_power / snr_linear
        
#         # Generate and apply noise
#         noise = torch.randn_like(x) * torch.sqrt(noise_power)
#         new_data.x = x + noise
        
#         return new_data

# def generate_ood_from_folds(src_root, dest_root, snr_db):
#     """
#     Reads existing folds, applies noise to test sets, and saves OOD versions.
#     """
#     ood_transform = AddGaussianNoiseOOD(snr_db=snr_db)
    
#     # Identify folds (fold_0, fold_1, etc.)
#     folds = [f for f in os.listdir(src_root) if f.startswith("fold_")]
    
#     for fold in sorted(folds):
#         print(f"--- Processing {fold} ---")
#         test_src_path = os.path.join(src_root, fold, "test")
        
#         if not os.path.exists(test_src_path):
#             print(f"Skipping {fold}: Test directory not found.")
#             continue
            
#         # 1. Load the test dataset using your existing GraphDataset class
#         # We pass the transform so __getitem__ applies the noise
#         test_dataset = GraphDataset(test_src_path)
        
#         # 2. Generate the OOD samples
#         print(f"Applying {snr_db}dB Gaussian Noise to {len(test_dataset)} samples...")
#         ood_test_data = [ood_transform(test_dataset[i]) for i in range(len(test_dataset))]
        
#         # 3. Define save path
#         fold_dest_dir = os.path.join(dest_root, fold)
#         os.makedirs(fold_dest_dir, exist_ok=True)
#         save_path = os.path.join(fold_dest_dir, "test_data_ood.pt")
        
#         # 4. Save using your existing utility
#         save_data_list(ood_test_data, save_path)
#         print(f"Saved OOD test set to: {save_path}")

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--src_dir", type=str, required=True, 
#                         help="Path to existing 'saved_folds' directory")
#     parser.add_argument("--dest_dir", type=str, default="data/OOD_folds", 
#                         help="Where to save the noisy OOD data")
#     parser.add_argument("--snr", type=int, default=0, 
#                         help="Signal-to-Noice Ratio in dB. Lower is noisier. 0 is a strong OOD signal.")
    
#     args = parser.parse_args()
    
#     generate_ood_from_folds(
#         src_root=args.src_dir, 
#         dest_root=args.dest_dir, 
#         snr_db=args.snr
#     )
#     print("\nOOD Generation Complete.")


import os
import warnings
from argparse import ArgumentParser
import numpy as np
import lightning.pytorch as pl
from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



from utils.dataloader_utils import (
    GraphDataset,
    HDFDataset_Writer,
    HDFDatasetLoader,
    save_data_list,
)

warnings.filterwarnings(
    "ignore", ".*does not have many workers.*"
)  # DISABLED ON PURPOSE

parser = ArgumentParser()
parser.add_argument("--timestep", type=int, default=6)  # 6 seconds sample are constructed from ictal samples.
parser.add_argument("--ictal_overlap", type=int, default=0) # In the paper 5 seconds overlap is used.
parser.add_argument("--inter_overlap", type=int, default=0) 
parser.add_argument("--preictal_overlap", type=int, default=0)
parser.add_argument("--seizure_lookback", type=int, default=600) # this time is considered for preictal periods.
parser.add_argument("--buffer_time", type=int, default=15) # buffer time is to discard the sample before and after seizures
parser.add_argument("--sampling_freq", type=int, default=256)
parser.add_argument("--downsampling_freq", type=int, default=60)
parser.add_argument("--smote", action="store_true", default=False)
parser.add_argument("--undersample", action="store_true", default=False) 
parser.add_argument("--train_test_split", type=float, default=0.0)
parser.add_argument("--fft", action="store_true", default=False)
parser.add_argument("--mne_features", action="store_true", default=False)
parser.add_argument("--normalizing_period", type=str, default="interictal")
parser.add_argument("--connectivity_metric", type=str, default="plv")
parser.add_argument("--npy_data_dir", type=str, default="data/npy_data")
parser.add_argument("--cache_dir",type=str,default="data/cache")
parser.add_argument("--event_tables_dir", type=str, default="data/event_tables")
parser.add_argument("--fold_data_dir",type=str,default = "saved_folds_trial")
parser.add_argument("--use_ictal_periods", action="store_true", default=False)
parser.add_argument("--use_preictal_periods", action="store_true", default=False)
parser.add_argument("--use_interictal_periods", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_splits", type=int, default=10)

args = parser.parse_args()
TIMESTEP = args.timestep
PREICTAL_OVERLAP = args.preictal_overlap
INTER_OVERLAP = args.inter_overlap
ICTAL_OVERLAP = args.ictal_overlap
SMOTE_FLAG = args.smote
NPY_DATA_DIR = args.npy_data_dir
FOLD_DATA_DIR = args.fold_data_dir
EVENT_TABLES_DIR = args.event_tables_dir
UNDERSAMPLE = args.undersample
FFT = args.fft
MNE_FEATURES = args.mne_features
USED_CLASSES_DICT = {
    "ictal": args.use_ictal_periods,
    "interictal": args.use_interictal_periods,
    "preictal": args.use_preictal_periods,
}
SFREQ = args.sampling_freq
DOWNSAMPLING_F = args.downsampling_freq
TRAIN_VAL_SPLIT = args.train_test_split
SEIZURE_LOOKBACK = args.seizure_lookback

NORMALIZING_PERIOD = args.normalizing_period
CONNECTIVITY_METRIC = args.connectivity_metric
BUFFER_TIME = args.buffer_time
CACHE_DIR = args.cache_dir
SEED = args.seed
KFOLD_CVAL_MODE = True
N_SPLITS = args.n_splits
INITIAL_CONFIG = dict(
    timestep=TIMESTEP,
    inter_overlap=INTER_OVERLAP,
    preictal_overlap=PREICTAL_OVERLAP,
    ictal_overlap=ICTAL_OVERLAP,
    seizure_lookback=SEIZURE_LOOKBACK,
    buffer_time=BUFFER_TIME,
    normalizing_period=NORMALIZING_PERIOD,
    smote=SMOTE_FLAG,
    undersampling=UNDERSAMPLE,
    sampling_freq=SFREQ,
    downsampling_freq=DOWNSAMPLING_F,
    fft=FFT,
    mne_features=MNE_FEATURES,
    used_classes_dict=USED_CLASSES_DICT,
    train_val_split=TRAIN_VAL_SPLIT,
    connectivity_metric=CONNECTIVITY_METRIC,
    seed=SEED,
)

#Pre precossing and saving the data to GCP bucket.

def generate_and_save_kfold_splits():
    # """Generate and save train, val, and test splits for all folds."""
    # writer = HDFDataset_Writer(
    #     seizure_lookback=SEIZURE_LOOKBACK,
    #     buffer_time=BUFFER_TIME,
    #     sample_timestep=TIMESTEP,
    #     inter_overlap=INTER_OVERLAP,
    #     preictal_overlap=PREICTAL_OVERLAP,
    #     ictal_overlap=ICTAL_OVERLAP,
    #     downsample=DOWNSAMPLING_F,
    #     sampling_f=SFREQ,
    #     smote=SMOTE_FLAG,
    #     connectivity_metric=CONNECTIVITY_METRIC,
    #     npy_dataset_path=NPY_DATA_DIR,
    #     event_tables_path=EVENT_TABLES_DIR,
    #     cache_folder=CACHE_DIR,
    # )
    # cache_file_path = writer.get_dataset()

    # loader = HDFDatasetLoader(
    #     root=cache_file_path,
    #     train_val_split_ratio=TRAIN_VAL_SPLIT,
    #     loso_subject=None,
    #     sampling_f=DOWNSAMPLING_F,
    #     extract_features=MNE_FEATURES,
    #     fft=FFT,
    #     seed=SEED,
    #     used_classes_dict=USED_CLASSES_DICT,
    #     normalize_with=NORMALIZING_PERIOD,
    #     kfold_cval_mode=KFOLD_CVAL_MODE,
    # )

    # full_data_path = loader.get_datasets()[0]
    full_data_path = "/home/harshad03897/cloud_drive/cache/2026-03-27_07-49-47/1774598111791.617/train/"
    full_dataset = GraphDataset(full_data_path)
    
    label_array = np.array([data.y.item() for data in full_dataset]).reshape(-1, 1)
    patient_id_array = np.array(
        [data.patient_id.item() for data in full_dataset]
    ).reshape(-1, 1)
    class_labels_patient_labels = np.concatenate(
        [label_array, patient_id_array], axis=1
    )

    kfold = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(
        kfold.split(np.zeros(len(full_dataset)), class_labels_patient_labels)
    ):
        print(f"Saving Data for Fold {fold}...")
        train_labels = class_labels_patient_labels[train_idx]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        sub_train_idx, val_idx = next(splitter.split(train_idx, train_labels))
        
        train_dataset = [full_dataset[idx] for idx in sub_train_idx]
        valid_dataset = [full_dataset[idx] for idx in val_idx]
        test_data = [full_dataset[idx] for idx in test_idx]
        
        fold_dir = os.path.join(FOLD_DATA_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        save_data_list(train_dataset, os.path.join(fold_dir, "train", "train_data.pt"))
        save_data_list(valid_dataset, os.path.join(fold_dir, "val", "valid_data.pt"))
        save_data_list(test_data, os.path.join(fold_dir, "test", "test_data.pt"))
        
    print("All splits successfully saved.")

if __name__ == "__main__":
    generate_and_save_kfold_splits()