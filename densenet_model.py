"""Model architecture for Method 2's frozen DenseNet feature extractor."""

import keras


def build_densenet_model(*, weights: str | None = "imagenet"):
    """Build Method 2; use weights=None when restoring a saved checkpoint."""
    backbone = keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=(96, 96, 3),
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(96, 96, 3))
    x = keras.applications.densenet.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(
        1, activation="sigmoid", name="classification_head"
    )(x)
    return keras.Model(inputs, outputs)
