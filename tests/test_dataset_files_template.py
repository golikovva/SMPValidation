import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from libs.validation.datasets.shapefile import ShapefileSicDataset


class ShapefileSicFilesTemplateTests(unittest.TestCase):
    @staticmethod
    def _create_file(root, name):
        date_dir = Path(root) / "20260914"
        date_dir.mkdir(exist_ok=True)
        file = date_dir / name
        file.touch()
        return file

    def test_uses_dynamic_default_files_template(self):
        with TemporaryDirectory() as root, patch.object(
            ShapefileSicDataset,
            "_create_grid",
            return_value=None,
        ):
            expected_file = self._create_file(root, "rasterized_S_7km.npy")

            dataset = ShapefileSicDataset(root, resolution=7)

        self.assertEqual(dataset._files_template, "*/rasterized_S_7km.npy")
        self.assertEqual(
            dataset.dates_dict,
            {datetime.date(2026, 9, 14): [expected_file]},
        )

    def test_init_files_template_overrides_dynamic_default(self):
        with TemporaryDirectory() as root, patch.object(
            ShapefileSicDataset,
            "_create_grid",
            return_value=None,
        ):
            self._create_file(root, "rasterized_S_7km.npy")
            expected_file = self._create_file(root, "custom.custom")

            dataset = ShapefileSicDataset(
                root,
                resolution=7,
                files_template="*/*.custom",
            )

        self.assertEqual(dataset._files_template, "*/*.custom")
        self.assertEqual(
            dataset.dates_dict,
            {datetime.date(2026, 9, 14): [expected_file]},
        )


if __name__ == "__main__":
    unittest.main()
