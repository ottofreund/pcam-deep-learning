"""Exercise the shared pipeline using tiny, offline CPU training runs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import keras
import numpy as np
import tensorflow as tf

import train


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.test_device = SimpleNamespace(name="test-device", device_type="CPU")

    def test_both_methods_produce_verified_run_artifacts(self):
        images = np.zeros((2, 96, 96, 3), dtype=np.uint8)
        labels = np.array([0, 1], dtype=np.int64)
        batches = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        batches = batches.with_options(options)

        # Keep tests offline; ImageNet initialization is used in production.
        model_builder = train.build_model

        def offline_model(method, *, pretrained=True):
            return model_builder(method, pretrained=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            roots = {
                method: Path(temporary_directory) / method
                for method in train.METHODS
            }
            for method in train.METHODS:
                with self.subTest(method=method):
                    args = train.parse_args([
                        "--method", method, "--epochs", "1", "--batch-size", "2",
                        "--steps-per-epoch", "1", "--validation-steps", "1",
                    ])
                    # The production GPU guard remains enabled. Only hardware
                    # discovery is mocked so this integration test runs on CPU.
                    with (
                        patch.dict(train.DEFAULT_OUTPUT_ROOTS, roots),
                        patch.object(
                            train, "load_training_data", return_value=(batches, batches)
                        ),
                        patch.object(train, "build_model", side_effect=offline_model),
                        patch.object(
                            tf.config, "list_physical_devices",
                            return_value=[self.test_device],
                        ),
                        patch.object(
                            tf.config.experimental, "get_device_details", return_value={}
                        ),
                    ):
                        run_directory = train.run_training(args)

                    self.assertEqual(run_directory.parent, roots[method])
                    for filename in (
                        "epoch-01.weights.h5", "history.csv", "ranking.csv",
                        "training.log", "run.json",
                    ):
                        self.assertTrue((run_directory / filename).is_file(), filename)
                    metadata = json.loads((run_directory / "run.json").read_text())
                    self.assertEqual(metadata["status"], "complete")
                    self.assertEqual(metadata["method"], method)
                    self.assertEqual(
                        metadata["top_validation_checkpoint"], "epoch-01.weights.h5"
                    )
                    verification = metadata["checkpoint_verification"][0]
                    self.assertEqual(verification["examples"], 2)
                    self.assertTrue(verification["finite_probabilities"])
                    history = (run_directory / "history.csv").read_text()
                    for metric in ("loss", "accuracy", "precision", "recall", "val_loss"):
                        self.assertIn(metric, history)
                    keras.backend.clear_session()

    def test_failed_run_records_method_and_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            args = train.parse_args([
                "--method", "method2", "--output-root", str(output_root),
            ])
            with (
                patch.object(
                    tf.config, "list_physical_devices", return_value=[self.test_device]
                ),
                patch.object(
                    tf.config.experimental, "get_device_details", return_value={}
                ),
                patch.object(
                    train, "load_training_data", side_effect=ValueError("invalid dataset")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "invalid dataset"):
                    train.run_training(args)
            metadata_path, = output_root.glob("*/run.json")
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["method"], "method2")
            self.assertEqual(metadata["error"], "ValueError: invalid dataset")

    def test_training_requires_gpu_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "runs"
            args = train.parse_args(["--output-root", str(output_root)])
            with patch.object(tf.config, "list_physical_devices", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "No TensorFlow GPU"):
                    train.run_training(args)
            self.assertFalse(output_root.exists())

    def test_invalid_limits_are_rejected(self):
        for option in (
            "--epochs", "--batch-size", "--steps-per-epoch", "--validation-steps"
        ):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    train.validate_args(train.parse_args([option, "0"]))


if __name__ == "__main__":
    unittest.main()
