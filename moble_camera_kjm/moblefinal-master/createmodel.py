import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image


def read_n_preprocess(image):
    """_summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image


# 경로 바꿔야 함! 코랩에서 돌렸음


def create_model():
    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.applications.MobileNetV2(
    #             input_shape=(224, 224, 3), include_top=False
    #         ),
    #         tf.keras.layers.GlobalAveragePooling2D(),
    #         tf.keras.layers.Dense(1, activation="sigmoid"),
    #     ]
    # )
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # return model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


train_data_dir = "/content/gdrive/My Drive/Colab Notebooks/train_data"
val_data_dir = "/content/gdrive/My Drive/Colab Notebooks/val_data"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=read_n_preprocess,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=read_n_preprocess
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    classes=["nonperson", "person"],
    shuffle=True,
)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    classes=["nonperson", "person"],
    shuffle=True,
)

model = create_model()

final = model.fit(train_generator, epochs=100, validation_data=val_generator)

model.save("/content/gdrive/My Drive/Colab Notebooks/face_detection_model1.h5")
