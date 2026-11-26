# Automated Malaria Detection from Blood Smear Images (CNN)

A Convolutional Neural Network that classifies red blood cell images as **healthy (Uninfected)** or **malaria-infected (Parasitized)**, supporting faster and more accurate medical diagnostics.

## Problem statement
Biocon Ltd. is working on automated diagnosis of blood samples. Develop a CNN model to classify red blood cell images as healthy or malaria-infected.

## Contents

| File | Purpose |
|------|---------|
| `Malaria_Detection_Colab.ipynb` | Notebook to open in Google Colab (recommended) |
| `malaria_cnn.py` | Standalone script for a local machine (needs the Kaggle dataset) |
| `colab_malaria_cnn.py` | Same code as the notebook, as a plain Python file |
| `README.md` | This file |

## Run on Google Colab (easiest)

1. Go to <https://colab.research.google.com> → **File → Upload notebook** → choose `Malaria_Detection_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**.

The dataset downloads automatically from the official NIH website, so no Kaggle account is needed. If the download ever fails, download `cell_images.zip` from Kaggle, upload it to the Colab Files panel, and re-run. Training takes a few minutes on a GPU. At the end, `malaria_results.zip` (model + plots) downloads to your computer.

## Run locally (optional)

```bash
pip install tensorflow scikit-learn matplotlib numpy pillow
# download the dataset from Kaggle: iarunava/cell-images-for-detecting-malaria, unzip into ./data
python malaria_cnn.py --data data
python malaria_cnn.py --demo                 # pipeline test on synthetic images
python malaria_cnn.py --predict cell.png     # predict one image
```

## Dataset
NIH Malaria Cell Images: 27,558 segmented cell images, balanced between Parasitized and Uninfected. Images are resized to 64×64. Split: 70% train / 15% validation / 15% test.

## Model
- Data augmentation (flips, rotation, zoom), training only
- 3 blocks of 2× [Conv → BatchNorm → ReLU], then MaxPool and Dropout(0.25), with 32 / 64 / 128 filters
- Global average pooling → Dense(128) → Dropout(0.5) → Sigmoid output (probability of infection)
- Adam (lr 1e-3), binary cross-entropy, early stopping and learning-rate reduction on plateau

## Outputs
Saved in `outputs/`: `malaria_cnn.keras`, `training_curves.png`, `confusion_matrix.png`, `roc_curve.png`, `sample_predictions.png`. The notebook also prints precision, recall, F1 and ROC-AUC on the test set.

## Notes
- For recall of infected cells (fewer missed cases), you can lower the 0.5 decision threshold.
- This is an academic project and not a certified medical device.
