# DEEP-MutOnco


This repository contains code, data, configurations, and plots to reproduce the results of **DEEP-MutOnco: A Two-Stage Deep Learning Framework for Tumor Type Classification and Survival Prognosis**.
- [Abstract](#abstract)
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)


# Abstract
Precision oncology relies on linking tumor molecular profiles to clinical outcomes. We present **DEEP-MutOnco**, a two-stage method:

1. **Classification stage**  
   - Based on FT-Transformer with PPI and protein‐sequence embeddings  
   - Trained on AACR Project GENIE v14.0 (38 cancer types; 43 971 MSK-IMPACT samples)  
   - Outputs a 512-dimensional representation  

2. **Survival stage**  
   - Features undergo variance filtering (threshold 0.001), correlation filtering (|r|>0.95) with OS time/status, and Z-score normalization  
   - Fitted to Cox proportional hazards on MSK-CHORD (5 cancer types; 16 377 patients)  
   - Achieves **C-index = 0.74**  

SHAP analysis identifies known drivers (e.g., TP53, KRAS) as top contributors to both tasks, providing biological insights into model decisions.

---

# Overview
<div align=center>
<img src="https://github.com/yyj971117/DEEP_MutOnco/blob/main/Overview.jpg" height="600" width="800">
</div>
1. **Datasets**  
   - **AACR GENIE 14.0** for classification (38 tumor types; 43 971 samples)  
   - **MSK-CHORD (Nature 2024)** for survival (5 tumor types; 16 377 samples)  

2. **Features**  
   - Binary mutation/indel, CNV focal & arm-level calls, MSI-sensor, mutational signatures  
   - PPI network metrics: degree, PageRank, betweenness  
   - ESM2 embeddings + Euclidean/Cosine/Manhattan distances  

3. **Model Architecture**  
   - **FT-Transformer** backbone with Squeeze-Excitation & Gated Feature Selector  
   - **Focal Loss** to handle class imbalance  
   - Softmax outputs over 38 classes  

4. **Survival Analysis**  
   - Variance & correlation filtering, Z-score scaling  
   - **Cox Proportional Hazards** model  
   - Kaplan–Meier stratification and log-rank tests  

5. **Interpretability**  
   - **Kernel SHAP** for global and per-cancer feature importance  

---

# System Requirements
## Hardware requirements
`DEEP-MutOnco` requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 18.04

### Python Dependencies
`DEEP-MutOnco` mainly depends on the Python scientific stack.
```
numpy
scipy
torch
pytorch-lightning
scikit-learn
pandas
scanpy
anndata
tensorboardX
pytorchtools
```
For specific setting, please see <a href="https://github.com/yyj971117/DEEP_MutOnco/blob/main/environment.yml">requirement</a>.

# Installation Guide
```
$ git clone https://github.com/yyj971117/DEEP_MutOnco.git
$ conda env create -f environment.yml
$ conda activate DEEP_MutOnco
```
# Detailed tutorials with example datasets
`DEEP-MutOnco` is a deep learning framework that utilizes autoencoders and multi-head attention mechanisms to accurately identify interactions between transcription factors and genes and infer gene regulatory networks.

The example can be seen in the <a href="https://github.com/yyj971117/DEEP_MutOnco/blob/main/DEEP-MutOnco/program/main.py">main.py</a>.

# DEEP-MutOnco

DEEP-MutOnco Model

## Quick Start (Tested on Linux)

  * Clone DEEP-MutOnco repository
```
git clone https://github.com/yyj971117/DEEP_MutOnco.git
```
  * Go to DEEP-MutOnco repository
```
cd DEEP-MutOnco
```
  * Create conda environment
```
conda env create -f environment.yml
```
  * activate DEEP-MutOnco environment
```
conda activate DEEP-MutOnco
```
 * Extract model files
   > Before running the program, you need to extract the dataset files. Please follow the steps below based on your operating system：
   > If you are using a Linux system and bzip2 is not installed, you can install it with the following command:
```
sudo apt-get update
sudo apt-get install -y bzip2
```
   > Navigate to the data directory and use bunzip2 to extract all .bz2 files:
```
cd DEEP_MutOnco/dataset/data_class

# Extract all .bz2 files
for f in *.bz2; do
    bunzip2 -k "$f"  # Use -k to keep the original compressed file
done

```
  * Update file paths in the code
    > Before running the program, ensure that any file paths used in the code are correctly updated to match your own directory structure. This is important because the default paths in the code may not align with where you have stored the necessary files. To do this, locate the file paths in the Python scripts (e.g., in `main.py`) and modify them to reflect the actual locations of your files on your system.
    > For example, If the file path in the code is pd.read_csv('~/DEEP-MutOnco/dataset/data_class/labels_test.csv'), change it to match your own directory structure.
```
pd.read_csv('~/DEEP-MutOnco/dataset/data_class/labels_test.csv')
```
  * Run the program
```
python main.py
```
* Choose the Appropriate Script Based on the Dataset
    > The following scripts are designed to work with different types of datasets. Please make sure to select the correct script based on the dataset you want to analyze:
     * main.py: This script is designed for classification tasks using the dataset.
     * main_os.py: This script is designed for survival analysis using the dataset.
     * Make sure to use the correct script based on whether you're analyzing classification or survival data.

