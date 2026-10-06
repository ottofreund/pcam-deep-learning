"""Utilities for producing reproducible, inspectable Method 1 training runs."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def configure_reproducibility(tf: Any, keras: Any, seed: int) -> None:
    """Seed Keras and require deterministic TensorFlow operations."""
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def checkpoint_pattern(run_directory: Path) -> Path:
    """Return the Keras path template used to retain every epoch's weights."""
    return Path(run_directory) / "epoch-{epoch:02d}.weights.h5"


def build_training_callbacks(run_directory: Path, callbacks_module: Any) -> list[Any]:
    """Build callbacks that retain every checkpoint and the complete metric history."""
    run_directory = Path(run_directory)
    return [
        callbacks_module.ModelCheckpoint(
            filepath=str(checkpoint_pattern(run_directory)),
            save_weights_only=True,
            save_best_only=False,
            save_freq="epoch",
            verbose=1,
        ),
        callbacks_module.CSVLogger(str(run_directory / "history.csv")),
    ]


def create_run_directory(
    output_root: Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Create a timestamped run directory without reusing an existing directory."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")

    candidate = output_root / timestamp
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{timestamp}-{suffix:02d}"
        suffix += 1

    candidate.mkdir()
    return candidate


def _history_value(history: Mapping[str, Sequence[float]], key: str, index: int):
    values = history.get(key)
    if values is None or index >= len(values):
        return ""
    return float(values[index])


def write_epoch_ranking(
    history: Mapping[str, Sequence[float]],
    output_path: Path,
) -> list[dict[str, int | float | str]]:
    """Rank epochs by validation loss, validation accuracy, then epoch number."""
    if "val_loss" not in history or not history["val_loss"]:
        raise ValueError("Training history must contain at least one val_loss value")

    metric_names = list(history.keys())
    epoch_count = len(history["val_loss"])
    rows: list[dict[str, int | float | str]] = []
    for index in range(epoch_count):
        row: dict[str, int | float | str] = {
            "epoch": index + 1,
            "checkpoint": f"epoch-{index + 1:02d}.weights.h5",
        }
        for metric_name in metric_names:
            row[metric_name] = _history_value(history, metric_name, index)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["val_loss"]),
            -float(row.get("val_accuracy", 0.0) or 0.0),
            int(row["epoch"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    fieldnames = ["rank", "epoch", "checkpoint", *metric_names]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows
