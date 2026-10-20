import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pathovision.classifier import TileClassifier
from torchvision import models


class _BatchRecorder:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, batch):
        self.batch_sizes.append(batch.shape[0])
        return torch.tensor([[3.0, 1.0]], device=batch.device).repeat(batch.shape[0], 1)


class ClassifierBatchingTests(unittest.TestCase):
    def test_predictions_run_in_bounded_batches_and_keep_tile_order(self):
        classifier = TileClassifier.__new__(TileClassifier)
        classifier.device = torch.device("cpu")
        classifier.batch_size = 2
        classifier.classes = ["negative", "positive"]
        classifier.validation_metrics = None
        classifier.test_metrics = None
        classifier.transform = lambda image: torch.zeros(3, 8, 8)
        classifier.model = _BatchRecorder()
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        tiles = [
            {"x": 0, "y": 0, "width": 16, "height": 16},
            {"x": 16, "y": 0, "width": 16, "height": 16},
            {"x": 0, "y": 16, "width": 16, "height": 16},
        ]
        result = classifier.predict_tiles(rgb, tiles)
        self.assertEqual(classifier.model.batch_sizes, [2, 1])
        self.assertEqual([(p["x"], p["y"]) for p in result["tiles"]], [(0, 0), (16, 0), (0, 16)])
        self.assertTrue(all(p["label"] == "negative" for p in result["tiles"]))

        classifier.model.batch_sizes.clear()
        patch_stream = ((tile, rgb[tile["y"]:tile["y"] + tile["height"],
                                    tile["x"]:tile["x"] + tile["width"]]) for tile in tiles)
        streamed = classifier.predict_patches(patch_stream)
        self.assertEqual(classifier.model.batch_sizes, [2, 1])
        self.assertEqual([(p["x"], p["y"]) for p in streamed["tiles"]], [(0, 0), (16, 0), (0, 16)])

    def test_safe_checkpoint_load_and_reject_unsupported_architecture(self):
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        payload = {"state_dict": model.state_dict(), "classes": ["negative", "positive"],
                   "architecture": "resnet18", "validation_metrics": None}
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "valid.pt"
            torch.save(payload, checkpoint)
            loaded = TileClassifier(checkpoint, device="cpu")
            self.assertEqual(loaded.classes, ["negative", "positive"])
            payload["architecture"] = "unknown"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "Only resnet18"):
                TileClassifier(checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
