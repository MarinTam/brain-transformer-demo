# Local Anomaly Detection in Brain Activity Time-Series

This repository implements and evaluates a Transformer-based anomaly detection framework for multivariate brain activity time-series data (fMRI/EEG), based on the TranAD architecture.

The project focuses on detecting localized anomalies in noisy biomedical time-series data and constructing robust evaluation pipelines for brain activity analysis.

---

# Overview

Rather than only applying existing models, this project emphasizes:

- Data engineering for biomedical time-series
- Pseudo-anomaly generation
- Experimental design for anomaly evaluation
- Explainable analysis of ROI-level abnormalities

---

# Key Contributions

## ROI Time-Series Extraction

Built preprocessing pipelines for extracting time-series signals from 100 brain ROIs (Regions of Interest) using datasets such as:

- HCP (Human Connectome Project)
- SRPBS dataset

---

## Pseudo-Anomaly Injection

Implemented custom anomaly injection methods designed for brain activity data.

Supported anomaly types include:

- Temporal anomalies
- Spatial anomalies
- Localized signal perturbations

These synthetic anomalies are used to quantitatively evaluate model robustness and detection capability.

---

## Experimental Design

Designed and evaluated 10 experimental cases with varying:

- anomaly injection settings
- subject groups
- preprocessing conditions

### Evaluation Cases

- **Case 1 -- 7**
  Synthetic anomaly evaluation

- **Case 8 -- 10**
  Evaluation under subject-group-based realistic conditions

---

# Installation

Python 3.7+ is required.

```bash
pip install -r requirements.txt
```

---

# Usage

## 1. Preprocessing

Convert raw brain activity data into model-ready ROI time-series format.

```bash
python preprocess.py --dataset SRPBS_dataset
```

---

## 2. Training and Evaluation

Run model training and anomaly detection evaluation.

```bash
python main.py --dataset SRPBS_dataset --model TranAD --retrain
```

---

# Results

The framework evaluates anomaly detection performance using metrics such as:

- ROC-AUC
- Global anomaly score
- ROI-level anomaly trends

After training, visualization outputs are automatically generated, including:

- anomaly score plots
- performance curves
- ROI activity visualizations

---

# Repository Structure

```text
.
├── main.py
├── preprocess.py
├── src/
├── data_engineering/
├── figures/
└── README.md
```

## Main Components

### `main.py`
Main training and evaluation script.

### `src/`
Core implementation of TranAD and related models.

### `data_engineering/`

Includes preprocessing and anomaly-generation utilities:

- `extract_roi_timeseries.py`
- `pseudo_anomaly.py`
- `make_fake_anomaly.py`
- `subject_analysis.py`

---

# Technologies

- Python
- PyTorch
- Transformer-based anomaly detection
- Time-series analysis
- Biomedical signal processing

---

# Notes

Due to confidentiality and dataset usage agreements, raw fMRI/EEG datasets are not included in this repository.

This repository contains only the implementation framework and preprocessing pipeline.
