import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2
from PIL import Image

from pathovision.demo import make_synthetic_demo
from pathovision.imaging import MAX_ANALYSIS_DIMENSION, detect_tissue_regions, focus_laplacian_variance, load_rgb
from pathovision.metrics import classification_metrics
from pathovision.pipeline import analyze_image, analyze_rgb, encode_png, save_outputs
from pathovision.report import build_analysis_report, encode_analysis_report
from pathovision.wsi import MAX_WSI_TILE_ANALYSES, analyze_wsi


class PathoVisionTests(unittest.TestCase):
    def test_synthetic_demo_is_repeatable_and_analyzable(self):
        image = make_synthetic_demo(size=256, seed=5)
        self.assertEqual(image.shape, (256, 256, 3))
        self.assertTrue(np.array_equal(image, make_synthetic_demo(size=256, seed=5)))
        result = analyze_rgb(image, tile_size=64)
        self.assertGreater(result["tile_count"], 0)
        self.assertGreaterEqual(result["analysis_time_seconds"], 0.0)

    def test_connected_tissue_region_measurements(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[10:30, 12:42] = 255
        mask[70:90, 80:105] = 255
        regions = detect_tissue_regions(mask)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]["area_px2"], 600)
        self.assertEqual((regions[0]["x"], regions[0]["y"]), (12, 10))
        self.assertAlmostEqual(regions[0]["centroid_x"], 26.5)

    def test_focus_proxy_is_higher_on_sharp_detail_than_blurred_detail(self):
        checker = (np.indices((128, 128)).sum(axis=0) % 2 * 255).astype(np.uint8)
        sharp = np.repeat(checker[..., None], 3, axis=2)
        blurred = cv2.GaussianBlur(sharp, (11, 11), sigmaX=2.0)
        self.assertGreater(focus_laplacian_variance(sharp), focus_laplacian_variance(blurred))

    def test_run_report_is_json_serializable_and_omits_upload_identity(self):
        result = analyze_rgb(make_synthetic_demo(size=256, seed=2), tile_size=64)
        report = build_analysis_report(result, source="Upload image", tile_size=64)
        encoded = encode_analysis_report(report)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["image"], {"width_px": 256, "height_px": 256})
        self.assertFalse(decoded["classifier"]["checkpoint_loaded"])
        self.assertFalse(decoded["interpretation"]["clinical_use"])
        self.assertNotIn("filename", encoded.decode("utf-8").lower())
        self.assertNotIn(str(Path.cwd()), encoded.decode("utf-8"))

    def test_run_report_keeps_group_counts_but_omits_group_ids(self):
        result = analyze_rgb(make_synthetic_demo(size=256, seed=3), tile_size=64)
        private_group_id = "private-patient-code"
        classifier = SimpleNamespace(
            training_config={"dataset": "imagefolder", "group_manifest_used": True,
                             "group_counts": {"train": 8, "val": 2, "test": 2},
                             "group_id": private_group_id},
            image_size=224, classes=["class_a", "class_b"],
            validation_metrics=None, test_metrics=None,
        )
        encoded = encode_analysis_report(build_analysis_report(
            result, source="Upload image", tile_size=64, classifier=classifier))
        report = json.loads(encoded)
        self.assertEqual(report["classifier"]["training_config"]["group_counts"],
                         {"train": 8, "val": 2, "test": 2})
        self.assertTrue(report["classifier"]["training_config"]["group_manifest_used"])
        self.assertNotIn(private_group_id, encoded.decode("utf-8"))

    def test_classification_metrics(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["negative", "positive"])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["classes"], ["negative", "positive"])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["per_class"][1]["f1"], 0.8)

    def test_classification_metrics_reject_invalid_inputs(self):
        cases = [
            (([0], [] , ["negative"]), "same number"),
            (([-1], [0], ["negative"]), "between 0 and 0"),
            (([0], [1], ["negative"]), "between 0 and 0"),
            (([0.5], [0], ["negative"]), "integer class indices"),
            (([0], [0], []), "at least one class"),
            (([0], [0], [""]), "non-empty strings"),
            (([0], [0], ["same", "same"]), "must be unique"),
        ]
        for args, message in cases:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, message):
                classification_metrics(*args)

    def test_image_analysis_and_exports(self):
        image = np.full((192, 192, 3), 245, dtype=np.uint8)
        image[24:168, 24:168] = (205, 125, 180)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "synthetic.png"
            Image.fromarray(image).save(source)
            result = analyze_image(source, tile_size=64)
            self.assertEqual((result["width"], result["height"]), (192, 192))
            self.assertGreater(result["tile_count"], 0)
            self.assertIn("focus_laplacian_variance", result["tiles"][0])
            self.assertGreater(result["tissue_area_fraction"], 0)
            overlay, mask, heatmap = save_outputs(result, temp, "check")
            self.assertTrue(overlay.is_file())
            self.assertTrue(mask.is_file())
            self.assertTrue(heatmap.is_file())
            self.assertEqual(result["tile_heatmap"].shape, image.shape)
            self.assertTrue(encode_png(result["tile_heatmap"]).startswith(b"\x89PNG\r\n\x1a\n"))

    def test_large_raster_preview_is_bounded_and_preserves_aspect(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "large_preview.tiff"
            Image.new("RGB", (MAX_ANALYSIS_DIMENSION + 904, 100), color=(170, 80, 120)).save(source)
            rgb = load_rgb(source)
            self.assertEqual(rgb.shape[1], MAX_ANALYSIS_DIMENSION)
            self.assertLessEqual(abs(rgb.shape[0] / 100 - MAX_ANALYSIS_DIMENSION / (MAX_ANALYSIS_DIMENSION + 904)), 0.01)

    def test_wsi_reads_only_a_bounded_sample_of_level_zero_tiles(self):
        class FakeSlide:
            dimensions = (2048, 2048)
            level_count = 2
            properties = {
                "openslide.vendor": "aperio",
                "openslide.mpp-x": "0.5",
                "openslide.mpp-y": "0.5",
                "openslide.objective-power": "20",
            }

            def __init__(self):
                self.read_count = 0
                self.thumbnail = Image.new("RGB", (128, 128), color=(190, 90, 150))

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def get_thumbnail(self, _size):
                return self.thumbnail

            def read_region(self, _location, _level, size):
                self.read_count += 1
                return Image.new("RGB", size, color=(190, 90, 150))

        slide = FakeSlide()
        fake_openslide = SimpleNamespace(
            OpenSlide=lambda _path: slide,
            PROPERTY_NAME_VENDOR="openslide.vendor",
            PROPERTY_NAME_MPP_X="openslide.mpp-x",
            PROPERTY_NAME_MPP_Y="openslide.mpp-y",
            PROPERTY_NAME_OBJECTIVE_POWER="openslide.objective-power",
        )
        with patch.dict("sys.modules", {"openslide": fake_openslide}):
            result = analyze_wsi("fake.svs", tile_size=64)
        self.assertEqual(result["width"], 2048)
        self.assertGreaterEqual(result["analysis_time_seconds"], 0.0)
        self.assertEqual(result["candidate_tile_count"], 1024)
        self.assertEqual(result["tile_count"], MAX_WSI_TILE_ANALYSES)
        self.assertEqual(slide.read_count, MAX_WSI_TILE_ANALYSES)
        self.assertEqual(result["tiles"][0]["x"] % 64, 0)
        self.assertEqual(result["slide_metadata"]["vendor"], "aperio")
        self.assertEqual(result["slide_metadata"]["mpp_x"], 0.5)
        self.assertEqual(result["tiles"][0]["width_um"], 32.0)
        self.assertTrue(result["regions"])
        self.assertIn("area_um2", result["regions"][0])
        self.assertIn("focus_laplacian_variance", result["tiles"][0])
        self.assertEqual(result["overlay"].shape, (128, 128, 3))

    def test_png_encoding_rejects_unsupported_arrays(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            encode_png(np.zeros((4, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()




