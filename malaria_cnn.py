"""
Automated Malaria Detection from Blood Smear Images (CNN)
==========================================================
Classifies red blood cell images as  Uninfected (healthy)  or  Parasitized (malaria-infected).

SETUP
-----
    pip install tensorflow scikit-learn matplotlib numpy pillow

DATASET (NIH Malaria Cell Images, ~27,558 images)
-------------------------------------------------
Download from Kaggle: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
    kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
    unzip cell-images-for-detecting-malaria.zip -d data
The script auto-finds the folder that contains  Parasitized/  and  Uninfected/.

USAGE
-----
    python malaria_cnn.py --demo                         # quick test on synthetic images (no download needed)
    python malaria_cnn.py --data data                    # train + evaluate on the real dataset
    python malaria_cnn.py --data data --epochs 30        # more epochs
    python malaria_cnn.py --predict some_cell.png        # predict one image with the saved model
"""
import argparse
import os
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

SEED = 42
IMG_SIZE = 64
BATCH = 64
OUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUT_DIR, "malaria_cnn.keras")
CLASS_NAMES = ["Uninfected", "Parasitized"]  # label 0 / label 1 (1 = infected = positive class)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def find_dataset_root(base):
    for root, dirs, _ in os.walk(base):
        if "Parasitized" in dirs and "Uninfected" in dirs:
            return root
    raise FileNotFoundError(
        f"Could not find 'Parasitized' and 'Uninfected' folders under '{base}'. "
        "Download the dataset (see top of this file) or run with --demo.")


def list_files(root):
    paths, labels = [], []
    for label, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(root, cls)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, f))
                labels.append(label)
    return np.array(paths), np.array(labels)


def make_demo_dataset(root="demo_cells", n_per_class=300):
    """Generate simple synthetic 'cells' so the whole pipeline can be tested offline."""
    from PIL import Image, ImageDraw
    if os.path.isdir(os.path.join(root, "Parasitized")):
        return root
    rng = np.random.default_rng(SEED)
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(root, cls), exist_ok=True)
    for cls in CLASS_NAMES:
        for i in range(n_per_class):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (20, 20, 20))
            d = ImageDraw.Draw(img)
            r = int(rng.integers(22, 29))
            c = IMG_SIZE // 2
            d.ellipse([c - r, c - r, c + r, c + r], fill=(int(rng.integers(200, 240)), 130, 150))
            if cls == "Parasitized":
                for _ in range(int(rng.integers(1, 4))):
                    x, y = rng.integers(c - r // 2, c + r // 2, size=2)
                    s = int(rng.integers(3, 7))
                    d.ellipse([x - s, y - s, x + s, y + s], fill=(70, 20, 110))
            arr = np.array(img).astype(np.int16) + rng.integers(-12, 12, (IMG_SIZE, IMG_SIZE, 3))
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
                os.path.join(root, cls, f"{cls}_{i}.png"))
    return root


def _load(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, tf.cast(label, tf.float32)


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model():
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")  # only active during training

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(Parasitized)

    model = keras.Model(inputs, outputs, name="malaria_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")])
    return model


# ----------------------------------------------------------------------------
# Plots / evaluation
# ----------------------------------------------------------------------------
def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="val")
    ax[0].set(title="Accuracy", xlabel="Epoch")
    ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="val")
    ax[1].set(title="Loss", xlabel="Epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()


def evaluate(model, test_paths, test_labels):
    probs = model.predict(make_ds(test_paths, test_labels, training=False), verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    print("\n=== Test-set results ===")
    print(classification_report(test_labels, preds, target_names=CLASS_NAMES, digits=4))
    auc = roc_auc_score(test_labels, probs)
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(test_labels, preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(test_labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # sample predictions grid
    idx = np.random.choice(len(test_paths), size=min(12, len(test_paths)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for a, k in zip(axes.ravel(), idx):
        img = tf.io.decode_image(tf.io.read_file(test_paths[k]), channels=3,
                                 expand_animations=False).numpy()
        a.imshow(img)
        ok = preds[k] == test_labels[k]
        a.set_title(f"True: {CLASS_NAMES[test_labels[k]]}\nPred: {CLASS_NAMES[preds[k]]} ({probs[k]:.2f})",
                    fontsize=8, color="green" if ok else "red")
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sample_predictions.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# Train / predict
# ----------------------------------------------------------------------------
def train(data_dir, epochs):
    root = find_dataset_root(data_dir)
    paths, labels = list_files(root)
    print(f"Found {len(paths)} images  |  infected: {labels.sum()}  healthy: {(labels == 0).sum()}")

    # stratified 70 / 15 / 15 split
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30,
                                                stratify=labels, random_state=SEED)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50,
                                              stratify=y_tmp, random_state=SEED)
    print(f"train {len(p_tr)} | val {len(p_va)} | test {len(p_te)}")

    train_ds = make_ds(p_tr, y_tr, training=True)
    val_ds = make_ds(p_va, y_va, training=False)

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    plot_history(history)

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    evaluate(model, p_te, y_te)
    print(f"Plots saved in ./{OUT_DIR}/")


def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train the model first.")
    model = keras.models.load_model(MODEL_PATH)
    img, _ = _load(image_path, 0)
    prob = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0, 0])
    label = CLASS_NAMES[int(prob >= 0.5)]
    conf = prob if prob >= 0.5 else 1 - prob
    print(f"{image_path}: {label}  (P(infected) = {prob:.4f}, confidence = {conf:.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CNN malaria cell classifier")
    ap.add_argument("--data", default="data", help="folder containing Parasitized/ and Uninfected/")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--demo", action="store_true", help="use synthetic images to test the pipeline")
    ap.add_argument("--predict", metavar="IMAGE", help="predict a single image with the saved model")
    args = ap.parse_args()

    if args.predict:
        predict(args.predict)
    elif args.demo:
        train(make_demo_dataset(), epochs=min(args.epochs, 5))
    else:
        train(args.data, args.epochs)
