# %% [markdown]
# # Automated Malaria Detection from Blood Smear Images (CNN)
# Classifies red blood cell images as **Healthy (Uninfected)** or **Malaria-infected (Parasitized)**.
#
# **How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**.
# The dataset (NIH Malaria Cell Images, 27,558 images) downloads automatically. No Kaggle account needed.

# %%
import os, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import subprocess, zipfile
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

SEED = 42
IMG_SIZE = 64
BATCH = 64
EPOCHS = 20
CLASS_NAMES = ["Uninfected", "Parasitized"]   # label 0 / 1  (1 = infected = positive class)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

# %% [markdown]
# ## 1. Load the dataset (70% train / 15% validation / 15% test)

# %%
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

train_ds, val_ds, test_ds, n_test = get_data(BATCH)

# %%
# Look at a few samples
plt.figure(figsize=(10, 5))
for images, labels in val_ds.take(1):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(CLASS_NAMES[int(labels[i])], fontsize=9)
        plt.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Build the CNN

# %%
def build_model():
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")  # active only during training

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
    outputs = layers.Dense(1, activation="sigmoid")(x)   # P(Parasitized)

    model = keras.Model(inputs, outputs, name="malaria_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return model

model = build_model()
model.summary()

# %% [markdown]
# ## 3. Train

# %%
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, m in zip(ax, ["accuracy", "loss"]):
    a.plot(history.history[m], label="train"); a.plot(history.history["val_" + m], label="val")
    a.set(title=m.capitalize(), xlabel="Epoch"); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=150); plt.show()

model.save(f"{OUT_DIR}/malaria_cnn.keras")

# %% [markdown]
# ## 4. Evaluate on the held-out test set

# %%
y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)
y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
auc = roc_auc_score(y_true, y_prob)
print(f"ROC-AUC: {auc:.4f}")

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
       xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150); plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC Curve"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=150); plt.show()

# %%
# Sample predictions (green = correct, red = wrong)
images, labels = next(iter(test_ds.unbatch().shuffle(2000, seed=1).batch(12)))
probs = model.predict(images, verbose=0).ravel()
plt.figure(figsize=(10, 7))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    pred = int(probs[i] >= 0.5); true = int(labels[i])
    plt.title(f"True: {CLASS_NAMES[true]}\nPred: {CLASS_NAMES[pred]} ({probs[i]:.2f})",
              fontsize=8, color="green" if pred == true else "red")
    plt.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/sample_predictions.png", dpi=150); plt.show()

# %% [markdown]
# ## 5. Predict your own image (optional)
# Run this cell and upload a single blood-cell image.

# %%
try:
    from google.colab import files
    uploaded = files.upload()
    for fname in uploaded:
        img = tf.io.decode_image(tf.io.read_file(fname), channels=3, expand_animations=False)
        img = tf.cast(tf.image.resize(img, (IMG_SIZE, IMG_SIZE)), tf.float32)
        p = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0, 0])
        label = CLASS_NAMES[int(p >= 0.5)]
        print(f"{fname}: {label} | P(infected) = {p:.4f}")
        plt.imshow(img.numpy().astype("uint8")); plt.title(f"{label} ({p:.2f})"); plt.axis("off"); plt.show()
except ImportError:
    print("Not running in Colab. Skipping upload.")

# %% [markdown]
# ## 6. Download results
# Saves the model and all plots as a zip you can download for your report and PPT.

# %%
import shutil
shutil.make_archive("malaria_results", "zip", OUT_DIR)
try:
    from google.colab import files
    files.download("malaria_results.zip")
except ImportError:
    print("Saved malaria_results.zip")
