import csv
from pathlib import Path
import tempfile
import unittest

from pathovision.split_audit import validate_group_manifest


class GroupSplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.files = {
            "train": [self._image("train/a/one.jpg"), self._image("train/a/two.jpg")],
            "val": [self._image("val/a/three.jpg")],
            "test": [self._image("test/a/four.jpg")],
        }
        self.manifest = self.root / "groups.csv"

    def tearDown(self):
        self.temp.cleanup()

    def _image(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test image")
        return path

    def _write(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["split", "group_id", "relative_path"])
            writer.writerows(rows)

    def _expected(self):
        return {split: paths for split, paths in self.files.items()}

    def test_valid_disjoint_groups_returns_counts(self):
        self._write([
            ("train", "patient-a", "train/a/one.jpg"),
            ("train", "patient-a", "train/a/two.jpg"),
            ("val", "patient-b", "val/a/three.jpg"),
            ("test", "patient-c", "test/a/four.jpg"),
        ])
        self.assertEqual(validate_group_manifest(self.root, self.manifest, self._expected()),
                         {"train": 1, "val": 1, "test": 1})

    def test_group_leakage_error_does_not_disclose_identifier(self):
        secret_id = "sensitive-patient-id"
        self._write([
            ("train", secret_id, "train/a/one.jpg"),
            ("train", secret_id, "train/a/two.jpg"),
            ("val", secret_id, "val/a/three.jpg"),
            ("test", "patient-c", "test/a/four.jpg"),
        ])
        with self.assertRaisesRegex(ValueError, "1 patient/slide group") as caught:
            validate_group_manifest(self.root, self.manifest, self._expected())
        self.assertNotIn(secret_id, str(caught.exception))

    def test_requires_complete_dataset_coverage(self):
        self._write([("train", "patient-a", "train/a/one.jpg")])
        with self.assertRaisesRegex(ValueError, "missing 3 dataset image"):
            validate_group_manifest(self.root, self.manifest, self._expected())

    def test_rejects_duplicate_paths(self):
        self._write([
            ("train", "patient-a", "train/a/one.jpg"),
            ("train", "patient-a", "train/a/one.jpg"),
        ])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_group_manifest(self.root, self.manifest, self._expected())

    def test_rejects_unknown_split_and_path_traversal(self):
        self._write([("validation", "patient-a", "train/a/one.jpg")])
        with self.assertRaisesRegex(ValueError, "unavailable split"):
            validate_group_manifest(self.root, self.manifest, self._expected())
        self._write([("train", "patient-a", "../outside.jpg")])
        with self.assertRaisesRegex(ValueError, "escapes the data root"):
            validate_group_manifest(self.root, self.manifest, self._expected())


if __name__ == "__main__":
    unittest.main()
