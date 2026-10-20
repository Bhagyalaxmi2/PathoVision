"""Optional bounded-memory whole-slide image overview and tile sampling."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .imaging import MAX_ANALYSIS_DIMENSION, detect_tissue_regions, focus_laplacian_variance, tissue_mask

MAX_WSI_TILE_ANALYSES = 512


def openslide_available() -> bool:
    try:
        import openslide  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def is_whole_slide(path: str | Path) -> bool:
    """Detect supported pyramid slide formats when the optional reader is installed."""
    if not openslide_available():
        return False
    import openslide
    return openslide.OpenSlide.detect_format(str(path)) is not None


def analyze_wsi(path: str | Path, tile_size: int = 256, classifier: Any = None) -> dict[str, Any]:
    """Analyze a pyramid overview and a spatially distributed, bounded set of level-0 tiles.

    Tile measurements and predictions use level-0 pixels. The overlay and heatmap are
    rendered on a low-resolution slide thumbnail. At most MAX_WSI_TILE_ANALYSES tiles
    are read from level 0, even when the slide contains many more tissue candidates.
    """
    started = perf_counter()
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    try:
        import openslide
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Whole-slide support is optional. Install it with `pip install -r requirements-wsi.txt`."
        ) from exc

    with openslide.OpenSlide(str(path)) as slide:
        width, height = slide.dimensions
        properties = slide.properties
        try:
            mpp_x = float(properties[openslide.PROPERTY_NAME_MPP_X])
            if not np.isfinite(mpp_x) or mpp_x <= 0:
                mpp_x = None
        except (KeyError, TypeError, ValueError):
            mpp_x = None
        try:
            mpp_y = float(properties[openslide.PROPERTY_NAME_MPP_Y])
            if not np.isfinite(mpp_y) or mpp_y <= 0:
                mpp_y = None
        except (KeyError, TypeError, ValueError):
            mpp_y = None
        slide_metadata = {
            "vendor": properties.get(openslide.PROPERTY_NAME_VENDOR),
            "objective_power": properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER),
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "pyramid_levels": int(slide.level_count),
        }
        thumbnail = slide.get_thumbnail((MAX_ANALYSIS_DIMENSION, MAX_ANALYSIS_DIMENSION)).convert("RGB")
        overview = np.asarray(thumbnail)
        overview_mask = tissue_mask(overview)
        regions = detect_tissue_regions(overview_mask)
        overview_height, overview_width = overview.shape[:2]
        scale_x, scale_y = width / overview_width, height / overview_height

        candidates = []
        for y in range(0, height, tile_size):
            tile_height = min(tile_size, height - y)
            top = min(int(y / scale_y), overview_height - 1)
            bottom = max(top + 1, min(int(np.ceil((y + tile_height) / scale_y)), overview_height))
            for x in range(0, width, tile_size):
                tile_width = min(tile_size, width - x)
                left = min(int(x / scale_x), overview_width - 1)
                right = max(left + 1, min(int(np.ceil((x + tile_width) / scale_x)), overview_width))
                coarse_fraction = float(np.mean(overview_mask[top:bottom, left:right] > 0))
                if coarse_fraction >= 0.03:
                    candidates.append({"x": x, "y": y, "width": tile_width, "height": tile_height})

        if len(candidates) > MAX_WSI_TILE_ANALYSES:
            indices = np.linspace(0, len(candidates) - 1, MAX_WSI_TILE_ANALYSES, dtype=np.int64)
            sampled = [candidates[index] for index in indices]
        else:
            sampled = candidates

        tiles = []
        for candidate in sampled:
            patch = np.asarray(slide.read_region(
                (candidate["x"], candidate["y"]), 0, (candidate["width"], candidate["height"])
            ).convert("RGB"))
            patch_mask = tissue_mask(patch)
            fraction = float(np.mean(patch_mask > 0))
            if fraction < 0.05:
                continue
            tile = {**candidate, "tissue_fraction": fraction,
                    "focus_laplacian_variance": focus_laplacian_variance(patch),
                    "mean_rgb": [float(value) for value in patch.reshape(-1, 3).mean(axis=0)]}
            if mpp_x is not None and mpp_y is not None:
                tile.update({"x_um": candidate["x"] * mpp_x, "y_um": candidate["y"] * mpp_y,
                             "width_um": candidate["width"] * mpp_x,
                             "height_um": candidate["height"] * mpp_y})
            tiles.append(tile)

        for region in regions:
            region.update({
                "x_level0": region["x"] * scale_x,
                "y_level0": region["y"] * scale_y,
                "width_level0": region["width"] * scale_x,
                "height_level0": region["height"] * scale_y,
                "area_level0_px2": region["area_px2"] * scale_x * scale_y,
            })
            if mpp_x is not None and mpp_y is not None:
                region["area_um2"] = region["area_level0_px2"] * mpp_x * mpp_y

        predictions = None
        if classifier is not None and tiles:
            crops = (
                (tile, np.asarray(slide.read_region(
                    (tile["x"], tile["y"]), 0, (tile["width"], tile["height"])
                ).convert("RGB")))
                for tile in tiles
            )
            predictions = classifier.predict_patches(crops)

        overlay = overview.copy()
        tint = np.zeros_like(overlay)
        tint[..., 1] = 190
        active = overview_mask > 0
        overlay[active] = (0.55 * overlay[active] + 0.45 * tint[active]).astype(np.uint8)
        for region in regions:
            x, y = region["x"], region["y"]
            cv2.rectangle(overlay, (x, y), (x + region["width"] - 1, y + region["height"] - 1), (255, 70, 30), 2)
        heatmap = np.full_like(overview, 255)
        for tile in tiles:
            x1 = int(tile["x"] / scale_x)
            y1 = int(tile["y"] / scale_y)
            x2 = min(overview_width, max(x1 + 1, int(np.ceil((tile["x"] + tile["width"]) / scale_x))))
            y2 = min(overview_height, max(y1 + 1, int(np.ceil((tile["y"] + tile["height"]) / scale_y))))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 210, 255), 1)
            color = cv2.applyColorMap(
                np.array([[round(255 * tile["tissue_fraction"])]], dtype=np.uint8), cv2.COLORMAP_TURBO
            )[0, 0]
            heatmap[y1:y2, x1:x2] = color[::-1]
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (35, 35, 35), 1)

        return {
            "width": int(width), "height": int(height), "thumbnail": overview,
            "mask": overview_mask, "overlay": overlay, "tile_heatmap": heatmap,
            "tiles": tiles, "tile_count": len(tiles), "candidate_tile_count": len(candidates),
            "regions": regions,
            "sample_limit": MAX_WSI_TILE_ANALYSES,
            "analysis_time_seconds": perf_counter() - started,
            "slide_metadata": slide_metadata,
            "tissue_area_fraction": float(np.mean(active)), "classifier": predictions,
            "analysis_type": "WSI thumbnail tissue mask + sampled level-0 tile analysis",
        }
