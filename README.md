# RUN PIPELINE


- Use **Python 3.10** with **PyTorch 2.1.0** and **PyTorch Geometric 2.4.0**.
- Create a Python virtual environment and install the required dependencies:
  
  ```bash
  pip install -r requirements.txt
  ```

* For GPU support, install the CUDA-enabled versions of **PyTorch** and **PyTorch Geometric** explicitly before running the project.

### 1. Download and preprocess the dataset

* Download the **CHB-MIT EEG dataset**.

* Run the initial preprocessing pipeline:

  ```bash
  python run_preprocessing.py
  ```

* Provide the following arguments:

  * `original_data_path`: path to the raw CHB-MIT dataset
  * `event_tables_path`: path to save seizure / non-seizure event tables for each patient and file
  * `preprocessed_edf_path`: kept for compatibility
  * `final_npy_path`: path to save the processed NumPy files
  * `annotation_files_path`: path to `records.txt`

### 2. Create the training dataset

* Generate the 10-fold dataset splits by running:

  ```bash
  python generate_dataset.py
  ```

* The default arguments can be used, or they can be modified as needed.

* This step creates:

  * train, validation, and test splits for each fold
  * a shared OOD dataset across folds

* All datasets are saved in `.pt` format.

### 3. Train the models

* Train the model on the in-distribution data:

  ```bash
  python train.py
  ```

* This script saves the model checkpoints after training.

### 4. Analyze the results

* Open `analysis-of-results.ipynb` to view:

  * classification performance
  * uncertainty metrics
  * explainability outputs
  * gated edge and feature importance results


# ARCHITECTURAL DESIGN

* **Preprocessed CHB-MIT EEG recordings** by standardizing electrode channels, filtering noise, and reducing artifacts so the data could be used consistently across patients.
* **Converted EEG signals into graph form** by treating electrodes as nodes and using PLV-based functional connectivity to define the edges.
* **Used handcrafted EEG features** such as spectral and temporal descriptors as node-level inputs to the graph model.
* **Applied a GATv2 model with 9 attention heads** to learn which brain connections and node features were most important for seizure classification.
* **Concatenated the attention heads** to build a richer graph embedding that captures multiple patterns of neural activity.
* **Used a global mean pooling readout** to convert node-level graph information into a single graph-level representation.
* **Built an MLP classification head** on top of the graph embedding to predict inter-ictal, pre-ictal, and ictal states.
* **Handled class imbalance with weighted cross-entropy**, so the model gave more attention to rare seizure samples.
* **Trained the model with AdamW and early stopping** to improve generalization and reduce overfitting.
* **Evaluated the model with 10-fold cross-validation** to check stability across different data splits.
* **Tested OOD generalization** by holding out specific patients and evaluating whether the model could handle unseen physiological patterns.
* **Used Monte Carlo Dropout at inference time** to estimate predictive uncertainty instead of producing only a single deterministic prediction.
* **Separated aleatoric and epistemic uncertainty**, so the system could distinguish noisy data from true model uncertainty.
* **Applied GNNExplainer and attention weights** to generate feature importance and edge importance explanations.
* **Added an uncertainty gate** so explanations are shown only when the prediction is confident and correct, reducing misleading visualizations.


<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/7e78a4a1-7726-474e-87c3-b2aabe3c8a1a" />

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/9b2f4435-4005-4fee-b672-32e304f22452" />

# RESULTS
* **Improved Out-of-Distribution Generalization:** While Monte Carlo Dropout introduced a modest reduction in ID performance, it improved OOD classification performance across unseen patients (CHB22–24) by an average of 4.35%, demonstrating better robustness to novel physiological patterns.
* **Meaningful Uncertainty Estimation:** Unlike the deterministic baseline, Monte Carlo Dropout successfully captured epistemic uncertainty, enabling the model to quantify its confidence when encountering unfamiliar patient data.
* **Better Calibration on Unseen Data:** Monte Carlo Dropout consistently improved uncertainty quality metrics across all OOD datasets, reducing Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), and Area Under the Risk-Coverage Curve (AURC) compared to the baseline model.
* **Detection of Distribution Shift:** Epistemic entropy increased noticeably on OOD patients while remaining relatively low on ID data, indicating that the uncertainty framework effectively recognized samples outside the training distribution.
* **More Reliable Connectivity Explanations:** Applying the uncertainty gate improved the consistency of GATv2 attention-based edge explanations. On CHB24, the Pearson correlation between explanations generated from correctly classified samples and explanations generated from all predictions increased by approximately 6%.
* **More Stable Feature Importance Explanations:** The uncertainty gate also improved the reliability of GNNExplainer feature importance masks. On CHB24, the Pearson correlation between correctly classified samples and all predicted samples improved by approximately 3%.

# ACKNOWLEDGEMENT:

This project utilizes code and methodologies adapted from the research "S. Mazurek, R. Blanco, J. Falcó-Roget and A. Crimi, "Explainable Graph Neural Networks for EEG Classification and Seizure Detection in Epileptic Patients," 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, pp. 1-5, doi: 10.1109/ISBI56570.2024.10635821.". And the corresponding repository https://github.com/szmazurek/sano_eeg. I am grateful to the authors for making their research and source code publicly available to the scientific community.
