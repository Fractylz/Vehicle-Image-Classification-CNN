# Imports
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Settings
DATASET_DIR = "data"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 26
CHECKPOINT_DIR = "checkpoints_customCNN"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Removing corrupted images
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

# Load Dataset
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
class_names = train_ds.class_names
print("Classes:", class_names)

# Data Augmentation
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=4)
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)


# Building the Model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for size in [32, 64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=len(class_names))
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)


# Load latest checkpoint
def latest_checkpoint(dir_path):
    files = [f for f in os.listdir(dir_path) if f.endswith(".keras")]
    if not files:
        return None
    files.sort()
    return os.path.join(dir_path, files[-1])


latest = latest_checkpoint(CHECKPOINT_DIR)
initial_epoch = 0
if latest:
    print(f"Resuming from checkpoint: {latest}")
    model = keras.models.load_model(latest)
    initial_epoch = int(os.path.basename(latest).split("-")[1].split(".")[0])

# Train from checkpoint
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "epoch-{epoch:02d}.keras"),
    save_weights_only=False,
    save_best_only=False,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_cb],
)

# Get true labels and predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Accuracy, Precision, Recall, F1
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


# Inference Function
def predict_image(model, img_path):
    img = keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)
    preds = model.predict(img_array)[0]
    for cls, prob in zip(class_names, preds):
        print(f"{cls}: {prob*100:.2f}%")
    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    print(f"\nPrediction: {class_names[pred_index]} ({confidence*100:.2f}%)")


# Testing
# predict_image(model, "data/Auto Rickshaws/Auto Rickshaw (1).jpg")
# predict_image(model, "data/Bikes/Bikes (1).jpg")
# predict_image(model, "data/Cars/Car (1).jpg")
# predict_image(model, "data/Motorcycles/Motorcycles (1).jpg")
# predict_image(model, "data/Planes/Planes (1).jpg")
# predict_image(model, "data/Ships/Ships (1).jpg")
# predict_image(model, "data/Trains/Trains (1).jpg")
