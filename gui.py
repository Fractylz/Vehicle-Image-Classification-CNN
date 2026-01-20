# vehicle_gui_mobilenet.py
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from keras import layers
from keras.applications import MobileNetV2
from keras.models import Model

# --- SETTINGS ---
IMAGE_SIZE = (224, 224)  # MobileNetV2 input
CLASS_NAMES = [
    "Auto Rickshaws",
    "Bikes",
    "Cars",
    "Motorcycles",
    "Planes",
    "Ships",
    "Trains",
]
WEIGHTS_PATH = "checkpoints_mnetv2/epoch-25.weights.h5"  # your MobileNetV2 weights file


# --- BUILD MODEL ARCHITECTURE ---
def build_model():
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs, outputs)
    return model


# --- LOAD MODEL AND WEIGHTS ---
model = build_model()
model.load_weights(WEIGHTS_PATH)
print("MobileNetV2 model loaded with weights!")


# --- GUI FUNCTIONS ---
def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if not file_path:
        return

    # Load image
    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    img_tk = ImageTk.PhotoImage(img_resized)

    # Display image
    canvas.img_tk = img_tk
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

    # Predict
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    probs = predictions[0]  # already softmax

    # Display probabilities
    prob_text = ""
    for cls, prob in zip(CLASS_NAMES, probs):
        prob_text += f"{cls}: {prob*100:.2f}%\n"
    pred_index = int(np.argmax(probs))
    confidence = float(np.max(probs))
    prob_text += f"\nPrediction: {CLASS_NAMES[pred_index]} ({confidence*100:.2f}%)"

    result_label.config(text=prob_text)


# --- GUI SETUP ---
root = tk.Tk()
root.title("Vehicle Classifier - MobileNetV2")

# Canvas for image
canvas = tk.Canvas(root, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1])
canvas.pack(padx=10, pady=10)

# Button to load image
btn_load = tk.Button(root, text="Upload Image", command=load_image)
btn_load.pack(pady=5)

# Label to show predictions
result_label = tk.Label(root, text="", justify="left", font=("Arial", 12))
result_label.pack(padx=10, pady=10)

# Run the GUI loop
root.mainloop()
