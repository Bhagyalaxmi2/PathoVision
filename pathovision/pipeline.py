from __future__ import annotations

from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .imaging import detect_tissue_regions, load_rgb, summarize_tiles, tissue_mask


def analyze_rgb(rgb: np.ndarray, tile_size: int = 256, classifier: Any = None) -> dict[str, Any]:
    """Analyze an in-memory RGB image."""
    started = perf_counter()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an 8-bit RGB image with shape (height, width, 3)")
    mask = tissue_mask(rgb)
    regions = detect_tissue_regions(mask)
    tiles = summarize_tiles(rgb, mask, tile_size)
    predictions = classifier.predict_tiles(rgb, tiles) if classifier is not None and tiles else None
    overlay = rgb.copy()
    tint = np.zeros_like(overlay)
    tint[..., 1] = 190
    active = mask > 0
    overlay[active] = (0.55 * overlay[active] + 0.45 * tint[active]).astype(np.uint8)
    for region in regions:
        x, y = region["x"], region["y"]
        cv2.rectangle(overlay, (x, y), (x + region["width"] - 1, y + region["height"] - 1), (255, 70, 30), 2)
    heatmap = np.full_like(rgb, 255)
    for tile in tiles:
        x, y, w, h = tile["x"], tile["y"], tile["width"], tile["height"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 210, 255), 1)
        color_bgr = cv2.applyColorMap(np.array([[round(255 * tile["tissue_fraction"])]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        heatmap[y:y + h, x:x + w] = color_bgr[::-1]
        cv2.rectangle(heatmap, (x, y), (x + w, y + h), (35, 35, 35), 1)
    return {
        "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "mask": mask,
        "overlay": overlay, "tile_heatmap": heatmap, "tiles": tiles,
        "regions": regions,
        "tissue_area_fraction": float(np.mean(active)),
        "tile_count": len(tiles), "classifier": predictions,
        "analysis_time_seconds": perf_counter() - started,
        "analysis_type": "heuristic tissue segmentation + connected-component region measurements" + (" + trained tile classifier" if predictions else ""),
    }


def analyze_image(path: str | Path, tile_size: int = 256, classifier: Any = None) -> dict[str, Any]:
    return analyze_rgb(load_rgb(path), tile_size, classifier)


def save_outputs(result: dict[str, Any], directory: str | Path, stem: str = "analysis") -> tuple[Path, Path, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    overlay_path, mask_path, heatmap_path = (out / f"{stem}_overlay.png", out / f"{stem}_mask.png", out / f"{stem}_tile_heatmap.png")
    Image.fromarray(result["overlay"]).save(overlay_path)
    Image.fromarray(result["mask"]).save(mask_path)
    Image.fromarray(result["tile_heatmap"]).save(heatmap_path)
    return overlay_path, mask_path, heatmap_path


def encode_png(image: np.ndarray) -> bytes:
    """Encode a uint8 image in memory for downloads without writing a file."""
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Expected a uint8 grayscale or color image")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()
