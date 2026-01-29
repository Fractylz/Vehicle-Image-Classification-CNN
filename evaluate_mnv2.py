import numpy as np
import keras
import matplotlib.pyplot as plt
from tensorflow import data as tf_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Settings
DATASET_DIR = "data"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "checkpoints_mnetv2/epoch-25.weights.h5"  # change if needed

# Load and rebuild model
from keras.applications import MobileNetV2
from keras import layers

base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=IMAGE_SIZE + (3,))
x = layers.Rescaling(1.0 / 255)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(7, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.load_weights(MODEL_PATH)
print("MobileNetV2 model loaded.")

# Load dataset
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

# Collect Prediction
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
print("\nOverall Accuracy:")
print(f"{accuracy_score(y_true, y_pred):.4f}")

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
plt.title("MobileNetV2 Confusion Matrix")
plt.tight_layout()
plt.show()
