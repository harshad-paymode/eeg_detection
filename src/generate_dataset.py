import os
import warnings
import shutil
from argparse import ArgumentParser
import numpy as np
import lightning.pytorch as pl
from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



from src.utils.dataloader_utils import (
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
parser.add_argument("--seizure_lookback", type=int, default=600) # this time window is considered for preictal periods.
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
    """Generate and save train, val, and test splits for all folds."""
    writer = HDFDataset_Writer(
        seizure_lookback=SEIZURE_LOOKBACK,
        buffer_time=BUFFER_TIME,
        sample_timestep=TIMESTEP,
        inter_overlap=INTER_OVERLAP,
        preictal_overlap=PREICTAL_OVERLAP,
        ictal_overlap=ICTAL_OVERLAP,
        downsample=DOWNSAMPLING_F,
        sampling_f=SFREQ,
        smote=SMOTE_FLAG,
        connectivity_metric=CONNECTIVITY_METRIC,
        npy_dataset_path=NPY_DATA_DIR,
        event_tables_path=EVENT_TABLES_DIR,
        cache_folder=CACHE_DIR,
    )
    cache_file_path = writer.get_dataset()

    loader = HDFDatasetLoader(
        root=cache_file_path,
        train_val_split_ratio=TRAIN_VAL_SPLIT,
        loso_subject=None,
        sampling_f=DOWNSAMPLING_F,
        extract_features=MNE_FEATURES,
        fft=FFT,
        seed=SEED,
        used_classes_dict=USED_CLASSES_DICT,
        normalize_with=NORMALIZING_PERIOD,
        kfold_cval_mode=KFOLD_CVAL_MODE,
    )

    full_data_path = loader.get_datasets()[0]

    #Seperate ID and OOD data
    id_dir = "data/id_data/"
    ood_dir = "data/ood_data/"  #this doesn't need to be divided furthere and will be used directly for testing
    os.makedirs(id_dir, exist_ok=True)
    os.makedirs(ood_dir, exist_ok=True)

    ood_patients = ["chb22.pt", "chb23.pt", "chb24.pt"]

    for file in os.listdir(full_data_path):
        if not file.endswith(".pt"):
            continue
            
        src_path = os.path.join(full_data_path, file)
        
        if file in ood_patients:
            shutil.move(src_path, os.path.join(ood_dir, file))
        else:
            shutil.move(src_path, os.path.join(id_dir, file))
            
    # override full_data_path to only point to the ID directory
    full_data_path = id_dir

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