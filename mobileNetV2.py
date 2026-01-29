# mobileNetV2_vehicle.py
import os
import numpy as np
import keras
from keras import layers
from keras.applications import MobileNetV2
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# --- SETTINGS ---
DATASET_DIR = "data"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 26
CHECKPOINT_DIR = "checkpoints_mnetv2"  # folder to save epoch checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 1️⃣ Remove corrupted images ---
num_skipped = 0
for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        fpath = os.path.join(class_path, fname)
        try:
            img = Image.open(fpath)
            img.verify()
        except Exception:
            num_skipped += 1
            os.remove(fpath)
print(f"Deleted {num_skipped} corrupted images.")

# --- 2️⃣ Load dataset ---
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
# Save class names immediately
class_names = train_ds.class_names
print("Classes:", class_names)

# --- 3️⃣ Data augmentation ---
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ]
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

# --- 4️⃣ Build model with transfer learning ---
base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False  # freeze pretrained layers

inputs = keras.Input(shape=IMAGE_SIZE + (3,))
x = layers.Rescaling(1.0 / 255)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# --- 5️⃣ Check if there is a previous checkpoint ---
latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
initial_epoch = 0
if latest:
    print(f"Resuming training from checkpoint: {latest}")
    model.load_weights(latest)
    # Extract epoch number from filename, assuming 'epoch-XX.h5' naming
    initial_epoch = int(latest.split("-")[-1].split(".")[0])

# --- 6️⃣ Training with checkpoints ---
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "epoch-{epoch:02d}.weights.h5"),
    save_weights_only=True,
    save_best_only=False,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_cb],
)


# --- 7️⃣ Inference function ---
def predict_image(model, img_path):
    img = keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # add batch dim

    preds = model.predict(img_array)[0]  # shape: (num_classes,)
    for cls, prob in zip(class_names, preds):
        print(f"{cls}: {prob*100:.2f}%")

    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    print(f"\nPrediction: {class_names[pred_index]} ({confidence*100:.2f}%)")


# --- 8️⃣ Example usage ---
predict_image(model, "data/Auto Rickshaws/Auto Rickshaw (1).jpg")
predict_image(model, "data/Bikes/Bikes (1).jpg")
predict_image(model, "data/Cars/Car (1).jpg")
predict_image(model, "data/Motorcycles/Motorcycles (1).jpg")
predict_image(model, "data/Planes/Planes (1).jpg")
predict_image(model, "data/Ships/Ships (1).jpg")
predict_image(model, "data/Trains/Trains (1).jpg")
