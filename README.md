# DEEP-MutOnco

## Abstract

### Background
Precision oncology requires integrated molecular profiling for accurate tumor classification and risk stratification. Current machine learning models often address these tasks separately, limiting their clinical utility.

### Model Introduction
**DEEP-MutOnco** is a deep learning multi-task framework that simultaneously:
- Integrates **multi-omics data**
- Classifies **38 tumor types** 
- Stratifies **survival risk**

| Feature | Description |
|---------|-------------|
| **Multi-task Architecture** | Joint classification + survival analysis |
| **Attention Mechanism** | Multi-head self-attention modules |
| **Interpretability** | SHAP analysis for feature importance |
| **Class Imbalance Handling** | Focal Loss implementation |

#### Two-stage Strategy

**Stage 1: Tumor Classification**
- Based on FT-Transformer with protein-protein interactions and protein‐sequence embeddings; Outputs a 512-dimensional representation
- The MSK-IMPACT dataset was split into 37,000 samples for model training and 6,971 for independent testing (AACR Project GENIE v14.0)
- Performance: 88.3% average accuracy (38 tumor types)

**Stage 2: Survival Risk Stratification**
- The MSK-CHORD 2024 cohort with overall survival follow-up data was separated into 13,225 samples for model training and 3,153 for independent validation
- Features undergo variance filtering (threshold 0.001), correlation filtering (|r|>0.95) with OS time/status, and Z-score normalization  
- Model: Cox proportional hazards
- Performance: C-index 0.74, significant risk stratification (log-rank p < 0.01), Kaplan-Meier curves clear separation between high- and low-risk groups

## Overview
<div align=center>
<img src="https://github.com/yyj971117/DEEP_MutOnco/blob/main/Overview.jpg" height="800"  width="1000">
</div>

**Figure 1. Schematic overview of the DEEP-MutOnco model workflow for multi-omics tumor analysis.** 

1. **Data processing and feature selection**  
   - Binary mutation/indel, CNV focal & arm-level calls, MSI-sensor, mutational signatures  
   - PPI network metrics: degree, PageRank, betweenness  
   - ESM2 embeddings + Euclidean/Cosine/Manhattan distances
   - Z-score normalization
   - Gated Feature Selector

2. **Model Architecture**  
   - **FT-Transformer** backbone with multi-head self-attention modules, followed by feed-forward neural network components, Squeeze-and-Excitation and residual normalization
   - **Focal Loss** to handle class imbalance
   - Softmax outputs over 38 classes  

3. **Survival Analysis**  
   - Variance & correlation filtering, Z-score scaling  
   - **Cox Proportional Hazards** model  
   - Kaplan–Meier stratification and log-rank tests  

4. **Interpretability**  
   - **Kernel SHAP** for global and per-cancer feature importance  

---

## System Requirements

### Hardware Requirements
`DEEP-MutOnco` requires a standard computer with sufficient RAM to support in-memory operations.

### Software Requirements

#### OS Requirements
This package is supported for **Linux**. The package has been tested on the following systems:
- Linux: Ubuntu 18.04 (CUDA 11.8)
- Linux: Ubuntu 22.04 (CUDA 12.4)

#### Python Dependencies
`DEEP-MutOnco` depends on the Python scientific stack:
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
optuna
shap
lifelines
networkx
```
For detailed requirements, see the <a href="https://github.com/yyj971117/DEEP_MutOnco/blob/main/environment.yml">environment files</a>.

# Installation and Usage

## Installation Guide

1. **Clone the repository**
```bash
git clone https://github.com/yyj971117/DEEP_MutOnco.git
cd DEEP-MutOnco
```

2. **Create conda environment**
```bash
# For Ubuntu 18.04 (CUDA 11.8)
conda env create -f environment_18.yml

# For Ubuntu 22.04 (CUDA 12.4)
conda env create -f environment_22.yml
```

3. **Activate the environment**
```bash
conda activate DEEP_MutOnco
```

## Usage

### Data Preparation

1. **Install bzip2**
```bash
sudo apt-get update
sudo apt-get install -y bzip2
```

2. **Extract dataset (.bz2 files)**
```bash
cd DEEP-MutOnco/dataset/data_class

for f in *.bz2; do
    bunzip2 -k "$f"
done

cd DEEP-MutOnco/dataset/data_os

for f in *.bz2; do
    bunzip2 -k "$f"
done
```

3. **Update file paths**
   > Update file paths in the Python scripts (e.g., lines 390-393, 396, 685-688, 692 in `main.py`; lines 633-636, 664 in `main_os.py`) to match your directory structure. For example, change `pd.read_csv('~/DEEP-MutOnco/dataset/data_class/labels_test.csv')` to reflect your actual file locations.

### Running the Model

Choose the appropriate script based on your analysis needs:

- **Tumor classification only**: `main.py`
  - Performs only **Stage 1 (tumor type classification)**
  - Outputs classification results for 38 tumor types
  
- **Complete pipeline**: `main_os.py`
  - Runs the full two-stage workflow
  - **Stage 1: Tumor classification**
  - **Stage 2: Survival risk stratification**

### Example Usage
```bash
# Test data loading
python -c "
import pandas as pd
df = pd.read_csv('DEEP-MutOnco/dataset/data_class/labels_test.csv')
print(f'Dataset loaded successfully: {df.shape}')
"

# For tumor classification only
python main.py
# For complete pipeline (classification + survival analysis)
python main_os.py
```

## Output

The model will generate:
- **Classification results**: Tumor type predictions with confidence scores
- **Survival analysis**: Risk stratification with hazard ratios and survival curves
- **Feature importance**: SHAP values for model interpretability

## Issues

If you encounter any problems, please open an issue on the GitHub repository.

