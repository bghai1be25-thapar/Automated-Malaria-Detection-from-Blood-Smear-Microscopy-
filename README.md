# Automated Malaria RBC Screening CNN (Biocon Ltd.)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning computer vision pipeline built for **Biocon Ltd.** to automate the screening of Giemsa-stained thin blood smear red blood cell (RBC) microscopy images into **Uninfected (Healthy)** and **Parasitized (Malaria-Infected)** classes.

Designed around **clinical sensitivity-first triage** and **Grad-CAM explainability**, the system assists clinical pathologists by flagging parasitized cells rapidly while routing borderline morphologies for human-in-the-loop verification.

---

## Key Highlights

* **Lightweight 4-Stage Residual-Inspired CNN (`Biocon_MalariaNet_v1`)**: Combines residual skip connections, Batch Normalization, Spatial Dropout, and dual Global Average + Global Max Pooling for fast edge/workstation inference.
* **Clinical Sensitivity-First Triage**: Uses cost-sensitive class weighting (`1.25x` penalty on missed parasitized cells) and a calibrated decision threshold (`0.42`) to minimize false negatives.
* **Pathologist Explainability (Grad-CAM)**: Automatically generates Gradient-weighted Class Activation Mapping overlays highlighting intra-erythrocytic *Plasmodium* chromatin dots and ring structures.
* **Self-Contained Verification Mode**: Includes a built-in synthetic Giemsa-stained RBC generator (`--mode demo`) to verify the full pipeline offline before downloading the full NIH dataset.

---

## Repository Structure

```text
Biocon-malaria-rbc-cnn/
├── biocon_malaria_cnn_monolith.py   # Complete end-to-end training, evaluation, Grad-CAM & CLI pipeline
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Standard Python, dataset & artifact exclusions
└── README.md                        # Project documentation & quick-start guide
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Offline Verification Demo (No Dataset Required)
Test the full training, evaluation dashboard, and Grad-CAM pipeline immediately using synthetic Giemsa-stained RBC tensors:
```bash
python biocon_malaria_cnn_monolith.py --mode demo
```

### 3. Train on the NIH Malaria Cell Images Dataset
Organize the [NIH NLM Malaria Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets) (27,558 cell images) into `Parasitized/` and `Uninfected/` subdirectories:
```text
cell_images/
├── Parasitized/
└── Uninfected/
```
Run full training with stratified `70% / 15% / 15%` splitting and microscopy-safe augmentations:
```bash
python biocon_malaria_cnn_monolith.py --mode train --data_dir ./cell_images --epochs 25 --batch_size 32
```

### 4. Single-Cell Clinical Inference & Grad-CAM Triage
Run inference on an individual cell image to receive a diagnostic probability, triage recommendation, and Grad-CAM overlay:
```bash
python biocon_malaria_cnn_monolith.py --mode predict --image_path ./sample_rbc.png
```

---

## Output Artifacts

All generated artifacts are saved automatically to `./biocon_malaria_artifacts/`:
* `biocon_malaria_best.keras` — Best model checkpoint selected via validation ROC-AUC.
* `clinical_evaluation_metrics.json` — Holdout test metrics (Sensitivity/Recall, Specificity, PPV, NPV, ROC-AUC, PR-AUC, Confusion Matrix counts).
* `biocon_diagnostic_dashboard.png` — 3-panel summary chart (Training/Validation AUC curves, Confusion Matrix, and ROC Curve).
* `gradcam_<image>.png` — Side-by-side comparison of the raw blood smear cell and Grad-CAM parasite localization heatmap.
* `training_log.csv` — Epoch-by-epoch training telemetry.

---

## Clinical Disclaimer
This software is intended as a clinical decision-support and research screening tool to assist qualified laboratory professionals and pathologists. All positive or borderline (`0.30 <= P(Parasitized) <= 0.60`) predictions must undergo confirmatory microscopic review per standard hematology protocols.
