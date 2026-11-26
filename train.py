"""
train.py: trains the malaria-cell CNN and saves the model, history and curves.

Works on Google Colab and locally. The dataset (NIH Malaria Cell Images, 27,558 images)
downloads automatically from the NIH website, so no Kaggle account is needed.

Colab:
    !python train.py --epochs 20

Local:
    pip install tensorflow scikit-learn matplotlib
    python train.py --epochs 20 --batch 64

Outputs (in ./outputs):
    malaria_cnn.keras    best model (lowest validation loss)
    history.json         per-epoch metrics
    training_curves.png  accuracy and loss plots
    test_split.txt       test-set size, for use by an evaluation script
"""
import argparse
import json
import os
import random
import subprocess
import zipfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

SEED = 42
IMG_SIZE = 64
OUT_DIR = "outputs"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


CLASS_FOLDERS = ["Uninfected", "Parasitized"]   # label 0 / 1  (1 = infected = positive class)
URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"   # official NIH source
DATA_DIR = "malaria_data"


def find_root(base):
    for r, dirs, _ in os.walk(base):
        if "Parasitized" in dirs and "Uninfected" in dirs:
            return r
    return None


def ensure_dataset():
    if find_root(DATA_DIR) is None:
        have_zip = os.path.exists("cell_images.zip") and zipfile.is_zipfile("cell_images.zip")
        if not have_zip:
            print("Downloading dataset from NIH (a few hundred MB)...")
            subprocess.run(["wget", "-q", "-O", "cell_images.zip", URL])
        if os.path.exists("cell_images.zip") and zipfile.is_zipfile("cell_images.zip"):
            with zipfile.ZipFile("cell_images.zip") as z:
                z.extractall(DATA_DIR)
    root = find_root(DATA_DIR)
    if root is None:
        raise FileNotFoundError(
            "Dataset not found and the automatic download failed. Download cell_images.zip manually "
            "(Kaggle: iarunava/cell-images-for-detecting-malaria, or the NIH link above), upload it to "
            "the Colab Files panel as 'cell_images.zip', then re-run this cell.")
    return root


def load_img(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, tf.cast(label, tf.float32)


def make_ds(p, y, batch, training):
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    if training:
        ds = ds.shuffle(len(p), seed=SEED)
    return ds.map(load_img, tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def get_data(batch):
    root = ensure_dataset()
    paths, labels = [], []
    for label, cls in enumerate(CLASS_FOLDERS):
        folder = os.path.join(root, cls)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, f))
                labels.append(label)
    paths, labels = np.array(paths), np.array(labels)

    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30,
                                                stratify=labels, random_state=SEED)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50,
                                              stratify=y_tmp, random_state=SEED)
    for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        print(f"{name:5s}: {len(y):6d} images | infected {int(y.sum()):6d} | healthy {int((y == 0).sum()):6d}")

    return (make_ds(p_tr, y_tr, batch, True), make_ds(p_va, y_va, batch, False),
            make_ds(p_te, y_te, batch, False), len(p_te))


def build_model(lr):
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="malaria_cnn")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return model


def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, m in zip(ax, ["accuracy", "loss"]):
        a.plot(history[m], label="train")
        a.plot(history["val_" + m], label="val")
        a.set(title=m.capitalize(), xlabel="Epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Train the malaria CNN")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

    train_ds, val_ds, _, n_test = get_data(args.batch)
    model = build_model(args.lr)
    model.summary()

    model_path = os.path.join(OUT_DIR, "malaria_cnn.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    with open(os.path.join(OUT_DIR, "test_split.txt"), "w") as f:
        f.write(f"{n_test}\n")
    plot_history(hist)

    best = int(np.argmin(hist["val_loss"]))
    print(f"\nBest epoch: {best + 1} | val_acc {hist['val_accuracy'][best]:.4f} "
          f"| val_auc {hist['val_auc'][best]:.4f}")
    print(f"Saved model and plots in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
