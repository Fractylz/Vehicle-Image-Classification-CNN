# predict.py

import numpy as np
import keras
import matplotlib.pyplot as plt

# --------------------------
# Settings
# --------------------------
image_size = (128, 128)
img_path = "data/Cars/Car (1).jpg"  # Change to any image you want

# --------------------------
# Load trained model
# --------------------------
model = keras.models.load_model("save_at_25.keras")
print("Model loaded!")

# --------------------------
# Load and preprocess image
# --------------------------
img = keras.utils.load_img(img_path, target_size=image_size)
plt.imshow(img)
plt.axis("off")
plt.show()

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Add batch axis

# --------------------------
# Predict
# --------------------------
predictions = model.predict(img_array)
probs = keras.ops.softmax(predictions[0]).numpy()

# --------------------------
# Get class names from training dataset
# --------------------------
# Note: manually define or load from train.py if needed
class_names = [
    "Auto Rickshaws",
    "Bikes",
    "Cars",
    "Motorcycles",
    "Planes",
    "Ships",
    "Trains",
]

print("Predicted probabilities:")
for cls, prob in zip(class_names, probs):
    print(f"{cls}: {prob*100:.2f}%")

pred_index = int(np.argmax(probs))
confidence = float(np.max(probs))
print(f"\nPrediction: {class_names[pred_index]} ({confidence*100:.2f}%)")
