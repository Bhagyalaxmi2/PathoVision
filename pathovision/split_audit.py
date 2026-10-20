"""Audit patient/slide-group separation for ImageFolder datasets."""
from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Iterable, Mapping


def validate_group_manifest(
    data_root: Path,
    manifest_path: Path,
    expected_paths: Mapping[str, Iterable[Path]],
) -> dict[str, int]:
    """Validate manifest coverage and ensure a patient/slide occurs in one split.

    The manifest must have ``split,group_id,relative_path`` columns. Only split
    group counts are returned; group identifiers are never included in output.
    """
    root = data_root.resolve()
    # Match normal CLI path behavior: a relative manifest is relative to the
    # process working directory, while its image paths are relative to data_root.
    manifest = manifest_path.resolve()
    if not manifest.is_file():
        raise ValueError("group manifest file does not exist")

    expected: dict[str, set[str]] = {}
    for split, paths in expected_paths.items():
        if split not in {"train", "val", "test"}:
            raise ValueError("expected split names must be train, val, or test")
        normalized = set()
        for path in paths:
            resolved = Path(path).resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError("dataset image path lies outside the data root") from exc
            normalized.add(relative.casefold())
        expected[split] = normalized

    seen_paths: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    group_counts: dict[str, set[str]] = {split: set() for split in expected}
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"split", "group_id", "relative_path"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("manifest requires split, group_id, and relative_path columns")
            for line_number, row in enumerate(reader, start=2):
                split = (row.get("split") or "").strip().lower()
                group = (row.get("group_id") or "").strip()
                rel_text = (row.get("relative_path") or "").strip()
                if not split or not group or not rel_text:
                    raise ValueError(f"manifest row {line_number} has an empty required field")
                if split not in expected:
                    raise ValueError(f"manifest row {line_number} uses an unavailable split")
                rel = Path(rel_text)
                if rel.is_absolute():
                    raise ValueError(f"manifest row {line_number} must use a relative image path")
                image_path = (root / rel).resolve()
                try:
                    normalized = image_path.relative_to(root).as_posix().casefold()
                except ValueError as exc:
                    raise ValueError(f"manifest row {line_number} escapes the data root") from exc
                if not image_path.is_file():
                    raise ValueError(f"manifest row {line_number} references a missing image")
                if normalized in seen_paths:
                    raise ValueError(f"manifest row {line_number} duplicates an image path")
                if normalized not in expected[split]:
                    raise ValueError(f"manifest row {line_number} does not match its ImageFolder split")
                seen_paths.add(normalized)
                group_splits.setdefault(group, set()).add(split)
                group_counts[split].add(group)
    except UnicodeError as exc:
        raise ValueError("group manifest must be UTF-8 CSV") from exc

    missing = sum(len(paths - seen_paths) for paths in expected.values())
    if missing:
        raise ValueError(f"group manifest is missing {missing} dataset image(s)")
    overlaps = sum(len(splits) > 1 for splits in group_splits.values())
    if overlaps:
        raise ValueError(f"group leakage detected: {overlaps} patient/slide group(s) occur in multiple splits")
    return {split: len(groups) for split, groups in group_counts.items()}
