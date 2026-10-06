import os

from datasets import load_dataset
import tensorflow as tf

from pcam_model import build_pcam_model

BATCH_SIZE = 64

DATASET = "1aurent/PatchCamelyon"

checkpoint_path = "training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
# The first run downloads the dataset and stores it in the Hugging Face cache.
ds_dict = load_dataset(DATASET)

train_set = ds_dict["train"].to_tf_dataset(columns="image", label_cols="label", batch_size=BATCH_SIZE, shuffle=True, drop_remainder=True)
val_set = ds_dict["valid"].to_tf_dataset(columns="image", label_cols="label", batch_size=BATCH_SIZE, shuffle=False, drop_remainder=True)

# Create model.
model = build_pcam_model()

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


