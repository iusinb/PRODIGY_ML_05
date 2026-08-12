"""
Train a food classifier using transfer learning (MobileNetV2 frozen base).
Designed to run reasonably fast on a CPU-only laptop by:
  - Using a lightweight base model (MobileNetV2)
  - Freezing the base model (only training a small head)
  - Using a small image size (128x128)

Usage:
    python train_model.py

Expects data in:
    ../data/train/<class_name>/*.jpg
    ../data/validation/<class_name>/*.jpg
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ---- Config ----
IMG_SIZE = (128, 128)
BATCH_SIZE = 16          # small batch size is fine and easier on CPU memory
EPOCHS = 10               # frozen-base training converges fairly quickly
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "food_classifier.keras")
CLASS_NAMES_OUT = os.path.join(os.path.dirname(__file__), "class_names.txt")


def build_data_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze - this is what keeps CPU training fast

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Couldn't find {TRAIN_DIR}. Download the Food-11 dataset and "
            f"arrange it as described in the README before running this script."
        )

    train_gen, val_gen = build_data_generators()
    num_classes = train_gen.num_classes
    print(f"Found {num_classes} classes: {list(train_gen.class_indices.keys())}")

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")

    # Save class names in index order so the app can map predictions back to labels
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(CLASS_NAMES_OUT, "w") as f:
        for i in range(num_classes):
            f.write(idx_to_class[i] + "\n")
    print(f"Saved class names to {CLASS_NAMES_OUT}")

    val_loss, val_acc = model.evaluate(val_gen)
    print(f"Final validation accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    main()
