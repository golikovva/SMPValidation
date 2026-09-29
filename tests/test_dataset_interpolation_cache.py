"""Exercise the real dataset constructors without optional native dependencies."""

import ast
from abc import ABC, abstractmethod
from contextlib import ExitStack
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def load_dataset_classes():
    """Load production class definitions, substituting only unused I/O imports."""
    namespace = {
        "__name__": "isolated_dataset_cache_tests",
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "Path": Path,
        "Interpolator": Mock(),
        "np": SimpleNamespace(load=Mock()),
        "xr": SimpleNamespace(open_dataset=Mock()),
        "open": mock_open(),
        "pickle": SimpleNamespace(load=Mock(return_value={})),
    }
    for filename in ("base", "amsr2", "cryosat", "shapefile", "nemo", "glorys", "stubs"):
        path = ROOT / "libs/validation/datasets" / f"{filename}.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name != "ConstantDataset"
        ]
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class DatasetInterpolationCacheTests(unittest.TestCase):
    def setUp(self):
        self.classes = load_dataset_classes()
        self.dataset_class = self.classes["Dataset"]
        self.interpolator = self.classes["Interpolator"]
        self.src_grid = object()
        self.dst_grid = object()

    def construct(self, dataset_class, args, **kwargs):
        # The constructors and interpolation hook remain the production code.
        # Only source-file discovery and geometry creation are unrelated here.
        with ExitStack() as stack:
            stack.enter_context(patch.object(dataset_class, "__abstractmethods__", set()))
            stack.enter_context(patch.object(dataset_class, "_create_dates_dict", return_value={}))
            stack.enter_context(patch.object(dataset_class, "_create_grid", return_value=self.src_grid))
            return dataset_class(*args, **kwargs)

    def test_cache_option_is_available_before_eager_hooks(self):
        cache_dir = Path("cached-weights")
        observed = []

        def dates(dataset):
            observed.append(dataset.interpolation_cache_dir)
            return {}

        def grid(dataset):
            observed.append(dataset.interpolation_cache_dir)
            return self.src_grid

        with patch.object(self.dataset_class, "__abstractmethods__", set()), patch.object(
            self.dataset_class, "_create_dates_dict", dates
        ), patch.object(self.dataset_class, "_create_grid", grid):
            dataset = self.dataset_class(
                "input", self.dst_grid, interpolation_cache_dir=cache_dir
            )

        self.assertEqual(observed, [cache_dir, cache_dir])
        self.interpolator.assert_called_once_with(self.src_grid, self.dst_grid)
        self.interpolator.return_value.initialize.assert_called_once_with(cache_dir=cache_dir)
        self.assertIs(dataset.interpolator, self.interpolator.return_value)

    def test_default_keeps_disk_cache_disabled(self):
        dataset = self.construct(
            self.dataset_class, ("input", self.dst_grid, [0, 2], "legacy-name"),
            files_template="*.custom",
        )

        self.assertIsNone(dataset.interpolation_cache_dir)
        self.assertEqual(dataset.path, Path("input"))
        self.assertEqual(dataset.average_times, [0, 2])
        self.assertEqual(dataset.name, "legacy-name")
        self.assertEqual(dataset._files_template, "*.custom")
        self.interpolator.return_value.initialize.assert_called_once_with(cache_dir=None)

    def test_no_destination_does_not_create_interpolator_or_touch_cache(self):
        cache_dir = ROOT / f"unused-interpolation-cache-{uuid4().hex}"
        with patch.object(Path, "mkdir") as mkdir, patch.object(Path, "open") as open_file:
            dataset = self.construct(
                self.dataset_class, ("input",), interpolation_cache_dir=cache_dir
            )

        self.assertIsNone(dataset.interpolator)
        self.interpolator.assert_not_called()
        mkdir.assert_not_called()
        open_file.assert_not_called()
        self.assertFalse(cache_dir.exists())

    def test_explicit_constructors_forward_option_and_preserve_positional_arguments(self):
        correction_model = object()
        datasets = [object()]
        variables = ["siconc"]
        lats_slice = slice(-100, None)
        cases = [
            ("Amsr2HSIDataset", ("input", self.dst_grid, [0], "name", "grid.nc"),
             {}, {"grid_file": "grid.nc"}),
            ("CryosatThickDataset", ("input", self.dst_grid, [0], "name", lats_slice),
             {}, {"lats_slice": lats_slice}),
            ("ShapefileSicDataset", ("input", 7, self.dst_grid, [0], "name"),
             {}, {"resolution": 7}),
            ("ShapefileDriftDataset", ("input", "arctic", self.dst_grid, [0], "name"),
             {}, {"region": "arctic"}),
            ("ShapefileThickDataset", ("input", 7, self.dst_grid, [0], "name"),
             {}, {"resolution": 7}),
            ("NemoThickDataset", ("input", self.dst_grid, [0], "name"),
             {"thickness_source": "sivolu"}, {"thickness_source": "sivolu"}),
            ("NemoGeneralIceDataset", ("input", variables, self.dst_grid, [0], "name", "mask"),
             {}, {"variables": variables, "mask_var": "mask"}),
            ("GlorysOperativeCorrectedSalinityDataset",
             ("input", self.dst_grid, [0], "name", correction_model),
             {}, {"correction_model": correction_model}),
            ("FusionDataset", (datasets, self.dst_grid, "name"),
             {}, {"datasets": datasets}),
        ]

        # Exercise both the original calls and opt-in calls, with str and Path.
        for class_name, args, extra, attributes in cases:
            for cache_dir in (None, "weights-cache", Path("weights-cache")):
                with self.subTest(dataset=class_name, cache_dir=cache_dir):
                    self.interpolator.reset_mock()
                    options = dict(extra, files_template="*.custom")
                    if cache_dir is not None:
                        options["interpolation_cache_dir"] = cache_dir
                    dataset_class = self.classes[class_name]
                    dataset = self.construct(dataset_class, args, **options)

                    self.assertEqual(dataset.interpolation_cache_dir, cache_dir)
                    self.assertIs(dataset.dst_grid, self.dst_grid)
                    self.assertEqual(dataset.name, "name")
                    self.assertEqual(dataset._files_template, "*.custom")
                    self.interpolator.return_value.initialize.assert_called_once_with(
                        cache_dir=cache_dir
                    )
                    for attribute, expected in attributes.items():
                        self.assertEqual(getattr(dataset, attribute), expected)

                    parameter = inspect.signature(dataset_class).parameters["interpolation_cache_dir"]
                    self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_inherited_dataset_constructor_accepts_cache_option(self):
        dataset = self.construct(
            self.classes["NemoSicDataset"], ("input", self.dst_grid),
            interpolation_cache_dir="weights-cache",
        )

        self.assertEqual(dataset.interpolation_cache_dir, "weights-cache")
        self.interpolator.return_value.initialize.assert_called_once_with(cache_dir="weights-cache")


if __name__ == "__main__":
    unittest.main()
