# test_predictions.py
import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.applications import MobileNetV2
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# ------------------------------
# SETTINGS
# ------------------------------
IMAGE_SIZE = (224, 224)
CHECKPOINT_PATH = "checkpoints/epoch-25.weights.h5"  # pick the checkpoint you want
DATASET_DIR = "data"
# ------------------------------

# --- 1️⃣ Rebuild model architecture (same as training) ---
base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = layers.Input(shape=IMAGE_SIZE + (3,))
x = layers.Rescaling(1.0 / 255)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
class_names = [
    cls
    for cls in sorted(os.listdir(DATASET_DIR))
    if os.path.isdir(os.path.join(DATASET_DIR, cls))
]
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = Model(inputs, outputs)

# --- 2️⃣ Load weights ---
model.load_weights(CHECKPOINT_PATH)
print(f"Loaded weights from: {CHECKPOINT_PATH}")


# --- 3️⃣ Inference function ---
def predict_image(model, img_path, class_names):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    preds = model.predict(img_array)[0]
    for cls, prob in zip(class_names, preds):
        print(f"{cls}: {prob*100:.2f}%")

    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    print(f"\nPrediction: {class_names[pred_index]} ({confidence*100:.2f}%)\n")


# --- 4️⃣ Loop through 1 example per class ---
for cls in class_names:
    class_folder = os.path.join(DATASET_DIR, cls)
    # pick the first image in the folder
    test_img = next(
        (
            f
            for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ),
        None,
    )
    if test_img:
        print(f"Testing class: {cls}")
        img_path = os.path.join(class_folder, test_img)
        predict_image(model, img_path, class_names)
