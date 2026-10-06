"""Train Method 1 and retain a separate weights file for every epoch."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from pcam_model import build_pcam_model
from training_utils import (
    build_training_callbacks,
    configure_reproducibility,
    create_run_directory,
    write_epoch_ranking,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DATASET = "1aurent/PatchCamelyon"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "training_1" / "runs"
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_SEED = 42


class TeeStream:
    """Write training output to both the terminal and a persistent log."""

    def __init__(self, terminal: TextIO, log: TextIO):
        self.terminal = terminal
        self.log = log

    def write(self, data: str) -> int:
        self.terminal.write(data)
        self.log.write(data)
        self.log.flush()
        return len(data)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Method 1 with one weights checkpoint per epoch."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Limit training steps; intended for smoke tests.",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=None,
        help="Limit validation steps; intended for smoke tests.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be at least 1")
    for name in ("steps_per_epoch", "validation_steps"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be at least 1")


def load_training_data(load_dataset: Any, batch_size: int):
    dataset = load_dataset(DATASET)
    train_set = dataset["train"].to_tf_dataset(
        columns="image",
        label_cols="label",
        batch_size=batch_size,
        shuffle=True,
        drop_remainder=True,
    )
    validation_set = dataset["valid"].to_tf_dataset(
        columns="image",
        label_cols="label",
        batch_size=batch_size,
        shuffle=False,
        drop_remainder=True,
    )
    return train_set, validation_set


def runtime_metadata(tf: Any, keras: Any, args: argparse.Namespace) -> dict[str, Any]:
    gpus = tf.config.list_physical_devices("GPU")
    gpu_details = []
    for device in gpus:
        details = tf.config.experimental.get_device_details(device)
        gpu_details.append(
            {
                "physical_device": device.name,
                "device_name": details.get("device_name", ""),
                "compute_capability": details.get("compute_capability", ""),
            }
        )
    return {
        "status": "running",
        "dataset": DATASET,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "validation_steps": args.validation_steps,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "gpu_devices": gpu_details,
        "selection_rule": "val_loss ascending, val_accuracy descending, epoch ascending",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def verify_checkpoints(
    run_directory: Path,
    epoch_count: int,
    validation_set: Any,
) -> list[dict[str, Any]]:
    images, _ = next(iter(validation_set.take(1)))
    results = []
    for epoch in range(1, epoch_count + 1):
        checkpoint = run_directory / f"epoch-{epoch:02d}.weights.h5"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing epoch checkpoint: {checkpoint}")
        model = build_pcam_model()
        model.load_weights(checkpoint)
        probabilities = np.asarray(model.predict_on_batch(images))
        if probabilities.shape[0] != len(images) or not np.isfinite(probabilities).all():
            raise ValueError(f"Checkpoint verification failed: {checkpoint}")
        results.append(
            {
                "epoch": epoch,
                "checkpoint": checkpoint.name,
                "examples": int(probabilities.shape[0]),
                "finite_probabilities": True,
            }
        )
    return results


def run_training(args: argparse.Namespace) -> Path:
    validate_args(args)

    import keras
    import tensorflow as tf
    from datasets import load_dataset

    configure_reproducibility(tf, keras, args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No TensorFlow GPU detected; refusing to fall back to CPU")

    run_directory = create_run_directory(args.output_root)
    metadata_path = run_directory / "run.json"
    metadata = runtime_metadata(tf, keras, args)
    metadata["run_directory"] = str(run_directory.resolve())
    write_json(metadata_path, metadata)

    start_time = time.monotonic()
    log_path = run_directory / "training.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            tee_stdout = TeeStream(sys.stdout, log_file)
            tee_stderr = TeeStream(sys.stderr, log_file)
            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                print(f"Run directory: {run_directory.resolve()}")
                print(f"TensorFlow GPUs: {[device.name for device in gpus]}")
                train_set, validation_set = load_training_data(
                    load_dataset, args.batch_size
                )

                model = build_pcam_model()
                model.compile(
                    optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy", "precision", "recall"],
                )
                callbacks = build_training_callbacks(
                    run_directory, tf.keras.callbacks
                )
                history = model.fit(
                    train_set,
                    validation_data=validation_set,
                    epochs=args.epochs,
                    callbacks=callbacks,
                    steps_per_epoch=args.steps_per_epoch,
                    validation_steps=args.validation_steps,
                    verbose=2,
                )

                ranking = write_epoch_ranking(
                    history.history, run_directory / "ranking.csv"
                )
                verification = verify_checkpoints(
                    run_directory, args.epochs, validation_set
                )
                print(
                    "Top validation checkpoint: "
                    f"{ranking[0]['checkpoint']} "
                    f"(val_loss={float(ranking[0]['val_loss']):.6f})"
                )

        metadata.update(
            {
                "status": "complete",
                "elapsed_seconds": time.monotonic() - start_time,
                "top_validation_checkpoint": ranking[0]["checkpoint"],
                "ranking": ranking,
                "checkpoint_verification": verification,
            }
        )
        write_json(metadata_path, metadata)
    except BaseException as error:
        metadata.update(
            {
                "status": "failed",
                "elapsed_seconds": time.monotonic() - start_time,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(metadata_path, metadata)
        raise

    return run_directory


def main() -> None:
    run_directory = run_training(parse_args())
    print(f"Completed run: {run_directory.resolve()}")


if __name__ == "__main__":
    main()
