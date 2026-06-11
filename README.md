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


# ACKNOWLEDGEMENT:

This project utilizes code and methodologies adapted from the research "S. Mazurek, R. Blanco, J. Falcó-Roget and A. Crimi, "Explainable Graph Neural Networks for EEG Classification and Seizure Detection in Epileptic Patients," 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, pp. 1-5, doi: 10.1109/ISBI56570.2024.10635821.". And the corresponding repository https://github.com/szmazurek/sano_eeg. I am grateful to the authors for making their research and source code publicly available to the scientific community.
