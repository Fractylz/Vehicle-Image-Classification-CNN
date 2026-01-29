import numpy as np
import keras
import matplotlib.pyplot as plt
from tensorflow import data as tf_data
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Settings
DATASET_DIR = "data"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load Model
model = keras.models.load_model("checkpoints_customCNN/save_at_25.keras")
print("Model loaded.")

# Load Dataset
_, val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = val_ds.class_names

val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

# Collect Predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Metrics
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
