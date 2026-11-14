"""
Model definitions. Both models take images as the loader produces them:
224x224x3, pixels 0 to 1. Model-specific scaling happens INSIDE the model,
so the loader and the app (UC-02) never need to change.
"""
import tensorflow as tf
from tensorflow.keras import layers


def build_baseline_cnn(num_classes, image_size=(224, 224)):
    """Small CNN trained from scratch. Gives a 'before transfer learning' number."""
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = inputs
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_mobilenetv2(num_classes, image_size=(224, 224), weights="imagenet"):
    """MobileNetV2 pretrained on ImageNet, with a new classification head.
    The pretrained base starts FROZEN, so only the new head learns at first.
    MobileNetV2 expects pixels from -1 to 1, so Rescaling converts 0-1 to -1-1."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False, weights=weights)
    base.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer"), base