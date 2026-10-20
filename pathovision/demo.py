"""Deterministic synthetic pattern for exercising the UI; not real tissue."""
import cv2
import numpy as np


def make_synthetic_demo(size: int = 640, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), (250, 247, 249), dtype=np.uint8)
    tissue = np.zeros((size, size), dtype=np.uint8)
    points = np.array([[70, 120], [155, 55], [300, 85], [470, 70], [570, 185],
                       [545, 360], [475, 520], [310, 565], [170, 520], [65, 390]], dtype=np.int32)
    scale = size / 640
    cv2.fillPoly(tissue, [np.round(points * scale).astype(np.int32)], 255)
    texture = rng.normal(0, 4, image.shape[:2]).astype(np.int16)
    pink = np.stack([np.full_like(texture, 225), np.full_like(texture, 157), np.full_like(texture, 192)], axis=2)
    image[tissue > 0] = np.clip(pink[tissue > 0] + texture[tissue > 0, None], 0, 255).astype(np.uint8)
    for _ in range(230):
        x, y = int(rng.integers(85, size - 75)), int(rng.integers(75, size - 75))
        if tissue[y, x] == 0:
            continue
        axes = (int(rng.integers(2, 6)), int(rng.integers(3, 8)))
        angle = int(rng.integers(0, 180))
        color = (int(rng.integers(90, 145)), int(rng.integers(45, 90)), int(rng.integers(120, 175)))
        cv2.ellipse(image, (x, y), axes, angle, 0, 360, color, -1)
    for cx, cy, radius in [(235, 255, 34), (405, 365, 42), (300, 455, 26)]:
        center = (int(cx * scale), int(cy * scale))
        cv2.circle(image, center, int(radius * scale), (250, 247, 249), -1)
        cv2.circle(image, center, int(radius * scale), (185, 100, 160), 2)
    return image
