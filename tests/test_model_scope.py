import unittest

from pathovision.model_scope import supports_high_resolution_tiles


class ModelScopeTests(unittest.TestCase):
    def test_pathmnist_benchmark_checkpoint_is_not_applied_to_high_resolution_tiles(self):
        self.assertFalse(supports_high_resolution_tiles({"dataset": "pathmnist"}))

    def test_custom_checkpoint_can_be_used_for_high_resolution_tiles(self):
        self.assertTrue(supports_high_resolution_tiles({"dataset": "imagefolder"}))

    def test_unknown_custom_checkpoint_metadata_remains_supported(self):
        self.assertTrue(supports_high_resolution_tiles({}))


if __name__ == "__main__":
    unittest.main()
