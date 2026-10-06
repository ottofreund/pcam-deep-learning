import os

from datasets import load_dataset
import tensorflow as tf
import keras

BATCH_SIZE = 64

DATASET = "1aurent/PatchCamelyon"

checkpoint_path = "training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
# The first run downloads the dataset and stores it in the Hugging Face cache.
ds_dict = load_dataset(DATASET)

train_set = ds_dict["train"].to_tf_dataset(columns="image", label_cols="label", batch_size=BATCH_SIZE, shuffle=True, drop_remainder=True)
val_set = ds_dict["valid"].to_tf_dataset(columns="image", label_cols="label", batch_size=BATCH_SIZE, shuffle=False, drop_remainder=True)
test_set = ds_dict["test"].to_tf_dataset(columns="image", label_cols="label", batch_size=BATCH_SIZE, shuffle=False, drop_remainder=True)

#Create model
model = keras.Sequential([
    keras.Input(shape=(96, 96, 3)),
    keras.layers.Rescaling(1.0 / 255), #pixels values to [0, 1]
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', data_format='channels_last'), #takes in a 96x96x3 image
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.35),
    keras.layers.Dense(1, activation='sigmoid')
])

# Callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

#train the model
history = model.fit(train_set, validation_data=val_set, epochs=10, callbacks=[cp_callback])

#model.evaluate(test_set)


