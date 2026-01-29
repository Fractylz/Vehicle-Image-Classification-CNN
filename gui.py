# vehicle_gui.py
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from keras import layers
from keras.models import load_model, Model
from keras.applications import MobileNetV2

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_SIZE_CNN = (128, 128)
IMAGE_SIZE_MNET = (224, 224)

CLASS_NAMES = [
    "Auto Rickshaws",
    "Bikes",
    "Cars",
    "Motorcycles",
    "Planes",
    "Ships",
    "Trains",
]

CUSTOM_CNN_PATH = "checkpoints_customCNN/save_at_25.keras"
MOBILENET_WEIGHTS = "checkpoints_mnetv2/epoch-25.weights.h5"


# ----------------------------
# MODEL BUILDERS
# ----------------------------
def load_custom_cnn():
    print("Loading Custom CNN...")
    return load_model(CUSTOM_CNN_PATH)


def build_mobilenet():
    print("Loading MobileNetV2...")
    base = MobileNetV2(
        input_shape=IMAGE_SIZE_MNET + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE_MNET + (3,))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs, outputs)
    model.load_weights(MOBILENET_WEIGHTS)
    return model


# ----------------------------
# LOAD DEFAULT MODEL
# ----------------------------
# current_model_name = tk.StringVar(value="MobileNetV2")
model = build_mobilenet()


# ----------------------------
# GUI LOGIC
# ----------------------------
def switch_model(*args):
    global model
    if current_model_name.get() == "Custom CNN":
        model = load_custom_cnn()
    else:
        model = build_mobilenet()
    result_label.config(text="Model switched.\nUpload an image.")


def predict_image():
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Choose image size
    img_size = (
        IMAGE_SIZE_MNET if current_model_name.get() == "MobileNetV2" else IMAGE_SIZE_CNN
    )

    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize(img_size)

    img_tk = ImageTk.PhotoImage(img_resized)
    canvas.config(width=img_size[0], height=img_size[1])
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk

    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    preds = model.predict(img_array)[0]

    # If Custom CNN → logits → softmax
    if current_model_name.get() == "Custom CNN":
        preds = tf.nn.softmax(preds).numpy()

    text = ""
    for cls, p in zip(CLASS_NAMES, preds):
        text += f"{cls}: {p*100:.2f}%\n"

    idx = int(np.argmax(preds))
    text += f"\nPrediction: {CLASS_NAMES[idx]} ({preds[idx]*100:.2f}%)"

    result_label.config(text=text)


# ----------------------------
# GUI SETUP
# ----------------------------
root = tk.Tk()
root.title("Vehicle Classifier")

current_model_name = tk.StringVar(root, value="MobileNetV2")

# Model selector
tk.Label(root, text="Select Model:", font=("Arial", 11)).pack()
dropdown = tk.OptionMenu(
    root, current_model_name, "MobileNetV2", "Custom CNN", command=switch_model
)
dropdown.pack(pady=5)

# Canvas
canvas = tk.Canvas(root, width=224, height=224)
canvas.pack(pady=10)

# Button
tk.Button(root, text="Upload Image", command=predict_image).pack(pady=5)

# Output
result_label = tk.Label(root, text="", justify="left", font=("Consolas", 11))
result_label.pack(padx=10, pady=10)

root.mainloop()
