"""Check shared evaluation metrics, CLI options, and saved model compatibility."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import eval as evaluation


class EvaluationTests(unittest.TestCase):
    def test_method_defaults_and_checkpoint_override(self):
        for method in evaluation.METHODS:
            with self.subTest(method=method):
                args = evaluation.parse_args(["--method", method])
                self.assertEqual(args.checkpoint, evaluation.DEFAULT_CHECKPOINTS[method])
                self.assertEqual(args.batch_size, 128)
                self.assertEqual(args.threshold, 0.5)
        args = evaluation.parse_args([
            "--method", "method2", "--checkpoint", "new-run/epoch-03.weights.h5",
            "--batch-size", "32", "--threshold", "0.75",
        ])
        self.assertEqual(args.checkpoint, Path("new-run/epoch-03.weights.h5"))
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.threshold, 0.75)

    def test_invalid_options_are_rejected(self):
        for options in (
            ["--method", "unknown"], ["--batch-size", "0"],
            ["--threshold", "-0.1"], ["--threshold", "1.1"],
            ["--threshold", "nan"],
        ):
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    evaluation.parse_args(options)

    def test_confusion_counts_metrics_and_threshold_across_batches(self):
        batches = [(np.zeros((3, 1)), [0, 1, 0]), (np.zeros((2, 1)), [1, 1])]
        for threshold, counts, accuracy in (
            (0.5, (2, 1, 1, 1), 0.6),
            (0.8, (2, 2, 0, 1), 0.8),
        ):
            with self.subTest(threshold=threshold):
                model = Mock()
                model.predict_on_batch.side_effect = [
                    np.array([[0.2], [0.8], [0.6]]), np.array([[0.9], [0.4]])
                ]
                results = evaluation.evaluate_batches(model, batches, threshold)
                tp, tn, fp, fn = counts
                self.assertEqual(results["examples"], 5)
                for key, expected in zip(
                    ("true_positives", "true_negatives", "false_positives", "false_negatives"),
                    counts,
                ):
                    self.assertEqual(results[key], expected)
                self.assertAlmostEqual(results["accuracy"], accuracy)
                self.assertAlmostEqual(results["precision"], tp / (tp + fp))
                self.assertAlmostEqual(results["recall"], tp / (tp + fn))
                self.assertAlmostEqual(results["f1"], 2 * tp / (2 * tp + fp + fn))

    def test_zero_positive_predictions_and_empty_data(self):
        model = Mock()
        model.predict_on_batch.return_value = np.zeros((2, 1))
        results = evaluation.evaluate_batches(model, [(np.zeros((2, 1)), [0, 0])])
        self.assertEqual(results["accuracy"], 1.0)
        for metric in ("precision", "recall", "f1"):
            self.assertEqual(results[metric], 0.0)
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluation.evaluate_batches(model, [])

    def test_invalid_predictions_and_labels_are_rejected(self):
        for predictions, labels in (
            ([np.nan], [0]), ([np.inf], [1]), ([0.5, 0.6], [1]),
            ([0.5], [2]), ([0.5], [0.5]), ([0.5], [256]),
        ):
            with self.subTest(predictions=predictions, labels=labels):
                model = Mock()
                model.predict_on_batch.return_value = np.array(predictions)
                with self.assertRaises(ValueError):
                    evaluation.evaluate_batches(model, [(np.zeros((1, 1)), labels)])

    def test_test_split_loader_retains_partial_batch(self):
        loader = Mock()
        batches = evaluation.load_test_batches(3, loader)
        loader.assert_called_once_with(evaluation.DATASET, split="test")
        loader.return_value.to_tf_dataset.assert_called_once_with(
            columns="image", label_cols="label", batch_size=3,
            shuffle=False, drop_remainder=False,
        )
        self.assertIs(batches, loader.return_value.to_tf_dataset.return_value)

    def test_missing_checkpoint_fails_before_building_model(self):
        with patch.object(evaluation, "build_model") as builder:
            with self.assertRaises(FileNotFoundError):
                evaluation.load_model("method2", Path("missing/epoch-01.weights.h5"))
            builder.assert_not_called()

    def test_cli_evaluates_existing_checkpoints_for_both_methods(self):
        # Use real saved weights and a final batch smaller than the requested size.
        images = np.zeros((3, 96, 96, 3), dtype=np.uint8)
        batches = [(images[:2], [0, 1]), (images[2:], [1])]
        for method in evaluation.METHODS:
            with self.subTest(method=method):
                output = io.StringIO()
                with (
                    patch("sys.argv", ["eval.py", "--method", method, "--batch-size", "2"]),
                    patch.object(evaluation, "load_test_batches", return_value=batches),
                    redirect_stdout(output),
                ):
                    evaluation.main()
                report = output.getvalue()
                self.assertIn(f"Method: {method}", report)
                self.assertIn("Examples: 3", report)
                self.assertIn("Confusion matrix:", report)
                for metric in ("Accuracy:", "Precision:", "Recall:", "F1:"):
                    self.assertIn(metric, report)


if __name__ == "__main__":
    unittest.main()
