"""Optional ResNet-18 tile classifier. A checkpoint must be trained by the user."""
from __future__ import annotations

from pathlib import Path
from itertools import islice
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


class TileClassifier:
    def __init__(self, checkpoint: str | Path, device: str | None = None, batch_size: int = 32):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = max(1, int(batch_size))
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must contain a mapping of model data")
        if payload.get("architecture", "resnet18") != "resnet18":
            raise ValueError("Only resnet18 checkpoints are supported")
        self.classes = payload.get("classes")
        if (not isinstance(self.classes, list) or not self.classes
                or not all(isinstance(name, str) and name for name in self.classes)
                or len(set(self.classes)) != len(self.classes)):
            raise ValueError("Checkpoint must contain a non-empty list of unique class names")
        self.validation_metrics = payload.get("validation_metrics")
        self.test_metrics = payload.get("test_metrics")
        self.training_config = payload.get("training_config", {})
        self.image_size = payload.get("image_size", 224)
        if not isinstance(self.image_size, int) or self.image_size < 16:
            raise ValueError("Checkpoint image_size must be an integer of at least 16")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint is missing a model state_dict")
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict_tiles(self, rgb: np.ndarray, tiles: list[dict[str, Any]]) -> dict[str, Any]:
        patches = (
            (tile, rgb[tile["y"]:tile["y"] + tile["height"], tile["x"]:tile["x"] + tile["width"]])
            for tile in tiles
        )
        return self.predict_patches(patches)

    @torch.inference_mode()
    def predict_patches(self, patches: Iterable[tuple[dict[str, Any], np.ndarray]]) -> dict[str, Any]:
        """Predict from a stream of (tile metadata, RGB crop) pairs in bounded batches."""
        outputs = []
        iterator = iter(patches)
        while chunk := list(islice(iterator, self.batch_size)):
            inputs = [self.transform(Image.fromarray(crop)) for _, crop in chunk]
            probs_batch = torch.softmax(self.model(torch.stack(inputs).to(self.device)), dim=1).cpu().numpy()
            for (tile, _), probs in zip(chunk, probs_batch):
                outputs.append({"x": tile["x"], "y": tile["y"], "label": self.classes[int(probs.argmax())],
                                "confidence": float(probs.max()), "probabilities": probs.tolist()})
        return {"classes": self.classes, "tiles": outputs,
                "mean_confidence": float(np.mean([p["confidence"] for p in outputs])) if outputs else None,
                "validation_metrics": self.validation_metrics, "test_metrics": self.test_metrics}
