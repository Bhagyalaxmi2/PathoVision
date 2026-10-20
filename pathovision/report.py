"""Privacy-conscious JSON run reports for PathoVision analyses."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


_TRAINING_FIELDS = (
    "dataset", "seed", "epochs_requested", "best_epoch", "batch_size",
    "train_samples", "train_samples_available", "validation_samples",
    "validation_samples_available", "test_samples", "test_samples_available",
    "selection_metric", "group_manifest_used", "group_counts",
)


def _metric_summary(metrics: dict[str, Any] | None) -> dict[str, float] | None:
    if not metrics:
        return None
    return {
        key: float(metrics[key])
        for key in ("accuracy", "macro_f1")
        if metrics.get(key) is not None
    }


def build_analysis_report(
    result: dict[str, Any],
    *,
    source: str,
    tile_size: int,
    classifier: Any = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an exportable summary without image pixels, upload names, or paths."""
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)

    training_config = getattr(classifier, "training_config", {}) if classifier else {}
    training_config = training_config if isinstance(training_config, dict) else {}
    checkpoint_loaded = classifier is not None
    model_used = result.get("classifier") is not None

    return {
        "schema_version": 1,
        "created_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "source_type": source,
        "image": {"width_px": int(result["width"]), "height_px": int(result["height"])},
        "analysis": {
            "method": result.get("analysis_type"),
            "tile_size_px": int(tile_size),
            "tissue_area_fraction": float(result["tissue_area_fraction"]),
            "candidate_tiles": result.get("candidate_tile_count"),
            "sampled_tiles": int(result["tile_count"]),
            "sample_limit": result.get("sample_limit"),
            "connected_tissue_regions": len(result.get("regions", [])),
            "elapsed_seconds": float(result["analysis_time_seconds"]),
            "slide_metadata": result.get("slide_metadata"),
        },
        "classifier": {
            "checkpoint_loaded": checkpoint_loaded,
            "inference_applied": model_used,
            "architecture": "ResNet-18" if checkpoint_loaded else None,
            "dataset": training_config.get("dataset") if checkpoint_loaded else None,
            "image_size_px": getattr(classifier, "image_size", None) if checkpoint_loaded else None,
            "class_names": list(getattr(classifier, "classes", [])) if checkpoint_loaded else [],
            "training_config": {key: training_config[key] for key in _TRAINING_FIELDS if key in training_config},
            "validation_metrics": _metric_summary(getattr(classifier, "validation_metrics", None)) if checkpoint_loaded else None,
            "held_out_test_metrics": _metric_summary(getattr(classifier, "test_metrics", None)) if checkpoint_loaded else None,
        },
        "interpretation": {
            "tissue_mask": "HSV saturation/brightness heuristic with morphology",
            "region_measurements": "Connected components of the heuristic tissue mask; not cell, lesion, or tumor detections.",
            "focus_metric": "Laplacian variance is scale- and content-dependent; no universal cutoff is applied.",
            "clinical_use": False,
        },
    }


def encode_analysis_report(report: dict[str, Any]) -> bytes:
    """Serialize a report as UTF-8 JSON and reject non-standard numeric values."""
    return (json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")
