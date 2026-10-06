import keras


def build_pcam_model():
    """Build the CNN architecture used to create the saved Method 1 weights."""
    return keras.Sequential(
        [
            keras.Input(shape=(96, 96, 3)),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
