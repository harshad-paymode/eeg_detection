# Data Pre-Processing Pipeline

**Dataset Characteristics (CHB-MIT):**
The CHB-MIT scalp EEG corpus is challenging to model because of severe class imbalance, strong inter-subject variability, and inconsistent montage configurations across long recording sessions. In particular, the ratio of inter-ictal to ictal samples is highly skewed, while seizure patterns vary significantly across patients in both spatial and temporal structure.

**Topological Unification:**
To keep node-level feature extraction consistent across different patient montages, the spatial features were standardized by selecting 18 common electrode channels. This preserved about 90% of the corpus and retained 178 of the 196 annotated ictal events.

**Signal Conditioning & Artifact Attenuation:**
EEG signals are highly sensitive to physiological and environmental artifacts such as ocular noise and powerline interference. To reduce high-frequency noise and lower computational cost, the raw 256 Hz signals were downsampled to 60 Hz. A 30 Hz low-pass filter was then applied along with a 50 Hz notch filter to remove AC powerline noise. Since the data was downsampled to 60 Hz, this 30 Hz cutoff served as the effective Nyquist limit, helping preserve alias-free signal representation.

**PCA-Based Denoising:**
To further reduce EMG artifacts and sensor-level hardware noise, Principal Component Analysis (PCA) was applied. Since the highest-variance components often capture broad artifact patterns, the first two principal components were removed. The 18 spatial signals were then reconstructed using the inverse transform of the remaining 16 components.

**Temporal Epoching & Class Stratification:**
Following clinical practice, the continuous EEG time series was divided into 6-second epochs. The data was organized into three physiological states: Inter-Ictal, Pre-Ictal, and Ictal. Pre-Ictal segments used a 15-second pre- and post-seizure buffer, while Ictal epochs used a 5-second sliding window overlap to naturally oversample the minority class and reduce class imbalance.

**Robust Standardization:**
Because EEG signals often show non-Gaussian distributions and heavy-tailed noise, a Robust Scaler was applied to each epoch. This method uses the median and interquartile range (IQR) instead of the mean and standard deviation, making it more resistant to extreme outliers and unusual signal variations.

**Static Graph Construction:**
To model spatial dependencies across the cerebral cortex in a static Graph Neural Network (GNN), an adjacency matrix was constructed using the Phase-Locking Value (PLV). This measured functional connectivity and phase synchronization between electrodes, allowing the brain’s electrical activity to be represented as a structured graph topology.

<img width="916" height="444" alt="image" src="https://github.com/user-attachments/assets/f75e102e-dda1-448d-9af0-5437cdf273a8" />


<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/7e78a4a1-7726-474e-87c3-b2aabe3c8a1a" />


<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/9b2f4435-4005-4fee-b672-32e304f22452" />



# ACKNOWLEDGEMENT:

This project utilizes code and methodologies adapted from the research "S. Mazurek, R. Blanco, J. Falcó-Roget and A. Crimi, "Explainable Graph Neural Networks for EEG Classification and Seizure Detection in Epileptic Patients," 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, pp. 1-5, doi: 10.1109/ISBI56570.2024.10635821.". And the corresponding repository https://github.com/szmazurek/sano_eeg. I am grateful to the authors for making their research and source code publicly available to the scientific community.
