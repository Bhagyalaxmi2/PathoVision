from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

MAX_ANALYSIS_DIMENSION = 4096


def load_rgb(path: str | Path) -> np.ndarray:
    """Load a raster image as RGB, limiting the longest edge for bounded processing.

    This is a preview workflow, not a streaming whole-slide-image reader. Tile
    coordinates are in the returned (possibly downsampled) image's pixel space.
    """
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.thumbnail((MAX_ANALYSIS_DIMENSION, MAX_ANALYSIS_DIMENSION))
            return np.asarray(image.convert("RGB"))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not safely read image: {path}. Very large whole-slide images need a streaming reader.") from exc


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Heuristic H&E-like tissue mask based on saturation and brightness."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    mask = ((sat > 18) & (val < 250)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def focus_laplacian_variance(rgb: np.ndarray) -> float:
    """Return a simple focus/edge-detail proxy for an RGB tile.

    This score is scale- and content-dependent; it is useful for relative review,
    not as a universal acceptance threshold or a validated focus classifier.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an 8-bit RGB image with shape (height, width, 3)")
    if min(rgb.shape[:2]) < 3:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_tissue_regions(mask: np.ndarray) -> list[dict[str, Any]]:
    """Measure connected components in the heuristic tissue mask.

    These are tissue-shaped connected regions, not cell, tumor, or lesion detections.
    Coordinates and area use the mask's pixel space.
    """
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Expected an 8-bit single-channel tissue mask")
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    minimum_area = max(64, int(round(mask.size * 0.0001)))
    regions = []
    for component_id in range(1, count):  # component zero is background
        x, y, width, height, area = (int(value) for value in stats[component_id])
        if area < minimum_area:
            continue
        regions.append({
            "region_id": len(regions) + 1,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "area_px2": area,
            "area_fraction": float(area / mask.size),
            "centroid_x": float(centroids[component_id, 0]),
            "centroid_y": float(centroids[component_id, 1]),
        })
    return regions


def summarize_tiles(rgb: np.ndarray, mask: np.ndarray, tile_size: int = 256) -> list[dict[str, Any]]:
    h, w = mask.shape
    results = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile_mask = mask[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            fraction = float(np.mean(tile_mask > 0)) if tile_mask.size else 0.0
            if fraction < 0.05:
                continue
            tile = rgb[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            results.append({"x": x, "y": y, "width": tile.shape[1], "height": tile.shape[0],
                            "tissue_fraction": fraction,
                            "focus_laplacian_variance": focus_laplacian_variance(tile),
                            "mean_rgb": [float(v) for v in tile.reshape(-1, 3).mean(axis=0)]})
    return results
