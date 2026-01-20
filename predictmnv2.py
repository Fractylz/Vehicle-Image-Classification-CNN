# predict_peek.py
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.applications import MobileNetV2
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------
IMAGE_SIZE = (224, 224)  # must match your model input size
WEIGHTS_PATH = "checkpoints/epoch-21.weights.h5"  # weights-only file
IMAGE_PATH = "data/Cars/Car (1).jpg"
CLASS_NAMES = [
    "Auto Rickshaws",
    "Bikes",
    "Cars",
    "Motorcycles",
    "Planes",
    "Ships",
    "Trains",
]
NUM_CLASSES = len(CLASS_NAMES)
# ------------------------------

# Rebuild model architecture (must match training)
base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = layers.Input(shape=IMAGE_SIZE + (3,))
x = layers.Rescaling(1.0 / 255)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation=None)(x)  # logits
model = Model(inputs, outputs)

# Load weights
model.load_weights(WEIGHTS_PATH)
print("Weights loaded from:", WEIGHTS_PATH)

# Optional: peek at weights of a layer
print("\n--- Sample Weights Peek ---")
for layer in model.layers:
    if len(layer.get_weights()) > 0:
        print(f"{layer.name}: {layer.get_weights()[0].shape}")
        break  # remove break to see more layers

# Load and preprocess image
img = load_img(IMAGE_PATH, target_size=IMAGE_SIZE)
plt.imshow(img)
plt.axis("off")
plt.show()

img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # add batch dimension

# Predict
predictions = model.predict(img_array)
probs = tf.nn.softmax(predictions[0]).numpy()  # convert logits to probabilities

# Print probabilities for all classes
print("\nPredicted probabilities:")
for cls, prob in zip(CLASS_NAMES, probs):
    print(f"{cls}: {prob*100:.2f}%")

# Print predicted class
pred_index = int(np.argmax(probs))
confidence = float(np.max(probs))
print(f"\nPrediction: {CLASS_NAMES[pred_index]} ({confidence*100:.2f}%)")
