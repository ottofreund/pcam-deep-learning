import os

from datasets import load_dataset
import tensorflow as tf
import keras

BATCH_SIZE = 64
DATASET = "1aurent/PatchCamelyon"

checkpoint_dir = "training_1"
os.makedirs(checkpoint_dir, exist_ok=True)

ds_dict = load_dataset(DATASET)

train_set = ds_dict["train"].to_tf_dataset(
    columns="image", label_cols="label",
    batch_size=BATCH_SIZE, shuffle=True, drop_remainder=True,
)
val_set = ds_dict["valid"].to_tf_dataset(
    columns="image", label_cols="label",
    batch_size=BATCH_SIZE, shuffle=False, drop_remainder=True,
)
test_set = ds_dict["test"].to_tf_dataset(
    columns="image", label_cols="label",
    batch_size=BATCH_SIZE, shuffle=False, drop_remainder=True,
)

# ImageNet-pretrained feature extractor. PCam images are already 96×96 RGB.
backbone = keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(96, 96, 3),
)
backbone.trainable = False

inputs = keras.Input(shape=(96, 96, 3))
x = keras.applications.densenet.preprocess_input(inputs)
x = backbone(x, training=False)  # keep BatchNorm in inference mode
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1, activation="sigmoid", name="classification_head")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
)

cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "cp-epoch-{epoch:02d}.weights.h5"),
    save_weights_only=True,
    save_freq="epoch",
    verbose=1,
)

history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=10,
    callbacks=[cp_callback],
)