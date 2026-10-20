import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardSmokeTests(unittest.TestCase):
    def test_default_dashboard_renders_without_exceptions(self):
        project_root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(project_root / "app.py"), default_timeout=30).run()

        self.assertFalse(app.exception)
        self.assertEqual([element.value for element in app.title], ["PathoVision"])
        self.assertIn(
            "Synthetic demo (not medical data)",
            app.radio[0].options,
        )
        self.assertEqual(app.radio[0].value, "Synthetic demo (not medical data)")
        self.assertGreaterEqual(len(app.metric), 4)


if __name__ == "__main__":
    unittest.main()
