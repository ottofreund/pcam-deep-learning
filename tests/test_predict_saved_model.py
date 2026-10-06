import unittest
from pathlib import Path

import numpy as np

from predict_saved_model import load_saved_model, load_validation_examples


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SavedModelPredictionTests(unittest.TestCase):
    def test_checkpoint_loads_and_predicts_a_small_batch(self):
        model = load_saved_model(REPOSITORY_ROOT / "training_1" / "cp.weights.h5")
        images = np.zeros((3, 96, 96, 3), dtype=np.uint8)

        probabilities = model.predict(images, verbose=0).reshape(-1)

        self.assertEqual(probabilities.shape, (3,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    def test_examples_are_loaded_from_validation_not_test(self):
        calls = []
        expected_examples = [{"image": object(), "label": 0}] * 3

        class FakeStreamingDataset:
            def take(self, count):
                calls.append(("take", count))
                return expected_examples

        def fake_load_dataset(dataset_name, *, split, streaming):
            calls.append((dataset_name, split, streaming))
            return FakeStreamingDataset()

        examples = load_validation_examples(3, load_dataset_fn=fake_load_dataset)

        self.assertEqual(examples, expected_examples)
        self.assertEqual(
            calls,
            [("1aurent/PatchCamelyon", "valid", True), ("take", 3)],
        )
        self.assertNotIn("test", calls[0][1])


if __name__ == "__main__":
    unittest.main()
