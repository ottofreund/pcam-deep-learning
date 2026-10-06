from datasets import load_dataset
import tensorflow as tf
import keras

BATCH_SIZE = 64
DATASET = "1aurent/PatchCamelyon"
WEIGHTS_PATH = "method2/cp-epoch-10.weights.h5"


def create_model():
    # The saved checkpoint restores the ImageNet-pretrained backbone weights.
    backbone = keras.applications.DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(96, 96, 3),
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(96, 96, 3))
    x = keras.applications.densenet.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


ds_dict = load_dataset(DATASET)

test_set = ds_dict["test"].to_tf_dataset(
    columns="image",
    label_cols="label",
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_remainder=False,
)

model = create_model()
model.load_weights(WEIGHTS_PATH)

all_labels = []
all_predictions = []

for images, labels in test_set:
    probabilities = model(images, training=False)
    predictions = tf.cast(probabilities >= 0.5, tf.int32)

    all_labels.append(tf.cast(tf.reshape(labels, [-1]), tf.int32))
    all_predictions.append(tf.reshape(predictions, [-1]))

labels = tf.concat(all_labels, axis=0)
predictions = tf.concat(all_predictions, axis=0)

confusion_matrix = tf.math.confusion_matrix(
    labels, predictions, num_classes=2
).numpy()

tn, fp = confusion_matrix[0]
fn, tp = confusion_matrix[1]

accuracy = (tp + tn) / confusion_matrix.sum()
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0.0
)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1_score:.4f}")
print("Confusion matrix:")
print(confusion_matrix)