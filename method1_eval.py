"""Evaluate the selected Model 1 checkpoint on the official PCam test set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent
DATASET = "1aurent/PatchCamelyon"
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "training_1" / "cp.weights.h5"


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT):
    """Rebuild Model 1 and load the selected epoch's weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    from pcam_model import build_pcam_model

    model = build_pcam_model()
    model.load_weights(checkpoint_path)
    return model


def load_test_batches(batch_size: int, load_dataset_fn=None):
    """Return every example in the official test split in deterministic order."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset

    test_data = load_dataset_fn(DATASET, split="test")
    return test_data.to_tf_dataset(
        columns="image",
        label_cols="label",
        batch_size=batch_size,
        shuffle=False,
        drop_remainder=False,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_batches(
    model: Any,
    batches: Iterable[tuple[Any, Any]],
    threshold: float = 0.5,
) -> dict[str, int | float]:
    """Evaluate binary predictions and return metrics plus confusion counts."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    true_positives = true_negatives = false_positives = false_negatives = 0

    for images, labels in batches:
        probabilities = np.asarray(model.predict_on_batch(images)).reshape(-1)
        labels = np.asarray(labels).astype(np.int8, copy=False).reshape(-1)

        if probabilities.size != labels.size:
            raise ValueError("The number of predictions does not match the labels")
        if not np.isfinite(probabilities).all():
            raise ValueError("Model predictions contain NaN or infinity")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Test labels must be binary (0 or 1)")

        predictions = probabilities >= threshold
        positive_labels = labels == 1
        true_positives += int(np.count_nonzero(predictions & positive_labels))
        true_negatives += int(np.count_nonzero(~predictions & ~positive_labels))
        false_positives += int(np.count_nonzero(predictions & ~positive_labels))
        false_negatives += int(np.count_nonzero(~predictions & positive_labels))

    examples = true_positives + true_negatives + false_positives + false_negatives
    if examples == 0:
        raise ValueError("The test dataset is empty")

    accuracy = _safe_ratio(true_positives + true_negatives, examples)
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(
        2 * true_positives,
        2 * true_positives + false_positives + false_negatives,
    )

    return {
        "examples": examples,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Model 1 weights on the official PCam test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model 1 weights (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size (default: 128)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Positive-class probability threshold (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batches = load_test_batches(args.batch_size)
    model = load_model(args.checkpoint)
    results = evaluate_batches(model, batches, args.threshold)

    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Dataset: {DATASET} (test split)")
    print(f"Examples: {results['examples']}")
    print(f"Threshold: {args.threshold:.3f}")
    print(
        "Confusion matrix: "
        f"TP={results['true_positives']} TN={results['true_negatives']} "
        f"FP={results['false_positives']} FN={results['false_negatives']}"
    )
    print(f"Accuracy:  {results['accuracy']:.6f}")
    print(f"Precision: {results['precision']:.6f}")
    print(f"Recall:    {results['recall']:.6f}")
    print(f"F1:        {results['f1']:.6f}")


if __name__ == "__main__":
    main()
