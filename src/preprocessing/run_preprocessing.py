from src.utils.utils import (
    get_annotation_files,
    preprocess_dataset_all,
    save_timeseries_array,
)
from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)    #Path to the raw/original data
    parser.add_argument("--event_tables_path", type=str, required=True)  #Path to save start and stope timestamps 
    parser.add_argument("--preprocessed_edf_path", type=str, required=True) #Path to save pre processed edf files of every recording for each patient
    parser.add_argument("--final_npy_path", type=str, required=True) #Saves every edf file as numpy file
    parser.add_argument("--annotation_files_path", type=str, required=True) #Path to record file which contatins names of seizure files for each patient.
    args = parser.parse_args()
    data_path = Path(args.data_path)
    event_tables_path = Path(args.event_tables_path)
    preprocessed_edf_path = Path(args.preprocessed_edf_path)
    final_npy_path = Path(args.final_npy_path)
    annotation_files_path = Path(args.annotation_files_path)
    # get_annotation_files(data_path, event_tables_path)
    # preprocess_dataset_all(
    #     annotation_files_path, data_path, preprocessed_edf_path
    # )
    save_timeseries_array(preprocessed_edf_path, final_npy_path)
