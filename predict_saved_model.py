import argparse
from pathlib import Path

import numpy as np


DATASET = "1aurent/PatchCamelyon"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "training_1" / "cp.weights.h5"


def load_saved_model(checkpoint_path=DEFAULT_CHECKPOINT):
    """Rebuild the Method 1 model and load its saved weights."""
    from pcam_model import build_pcam_model

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_pcam_model()
    model.load_weights(checkpoint_path)
    return model


def load_validation_examples(num_examples, load_dataset_fn=None):
    """Load only a small slice of validation data, never the official test split."""
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")

    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset

    validation_stream = load_dataset_fn(DATASET, split="valid", streaming=True)
    return list(validation_stream.take(num_examples))


def predict_examples(model, examples):
    images = np.stack(
        [np.asarray(example["image"].convert("RGB"), dtype=np.uint8) for example in examples]
    )
    labels = np.asarray([int(example["label"]) for example in examples])
    probabilities = model.predict(images, verbose=0).reshape(-1)
    return labels, probabilities


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test saved PCam weights on a few validation examples."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a Keras .weights.h5 checkpoint.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of validation examples to predict (default: 5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_validation_examples(args.num_examples)
    model = load_saved_model(args.checkpoint)
    labels, probabilities = predict_examples(model, examples)

    print(f"Loaded weights: {args.checkpoint}")
    print(f"Predictions on {len(labels)} validation examples (official test set not used):")
    for index, (label, probability) in enumerate(zip(labels, probabilities), start=1):
        predicted_label = int(probability >= 0.5)
        print(
            f"  {index}: true={label} predicted={predicted_label} "
            f"positive_probability={probability:.4f}"
        )


if __name__ == "__main__":
    main()
