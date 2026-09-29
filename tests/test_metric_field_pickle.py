"""Result metadata survives ndarray copying and new/legacy pickle files."""

import ast
import copy
from datetime import date
from pathlib import Path
import pickle
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "libs.validation.validator": "libs/validation/validator.py",
    "libs.validation.aggregators": "libs/validation/aggregators.py",
}


def metric_field_node(relative_path):
    """Read the real class without importing optional geospatial packages."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return next(node for node in tree.body if getattr(node, "name", None) == "MetricField")


def load_metric_field(relative_path, name):
    node = metric_field_node(relative_path)
    node.name = name
    # Register each class under a unique name in this importable test module,
    # allowing pickle to resolve its identity without importing the application.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / relative_path), "exec"), globals())
    return globals()[name]


ValidatorMetricField = load_metric_field(SOURCES["libs.validation.validator"], "ValidatorMetricField")
AggregatorMetricField = load_metric_field(SOURCES["libs.validation.aggregators"], "AggregatorMetricField")
FIELD_CLASSES = (ValidatorMetricField, AggregatorMetricField)


class MetricFieldPickleTests(unittest.TestCase):
    @staticmethod
    def field(field_class):
        return field_class(
            np.array([[1.0, np.nan], [3.5, 4.0]], dtype=np.float32, order="F"),
            buoy_ids=np.array(["300001", "300002"]),
            coords=np.array([[82.5, 179.5], [82.6, -179.5]]),
            time=np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[ns]"),
            nested={"channels": ("u", "v"), "quality": [np.array([True, False]), None]},
        )

    def assert_field_equal(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(set(actual.meta), set(expected.meta))
        for key in ("buoy_ids", "coords", "time"):
            np.testing.assert_array_equal(actual.meta[key], expected.meta[key])
        self.assertEqual(actual.meta["nested"]["channels"], expected.meta["nested"]["channels"])
        np.testing.assert_array_equal(
            actual.meta["nested"]["quality"][0], expected.meta["nested"]["quality"][0],
        )
        self.assertIsNone(actual.meta["nested"]["quality"][1])

    def test_roundtrip_nested_metadata_all_pickle_protocols(self):
        for field_class in FIELD_CLASSES:
            source = self.field(field_class)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(field=field_class.__name__, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(source, protocol=protocol))
                    self.assert_field_equal(restored, source)
                    self.assertTrue(restored.flags.f_contiguous)
                    self.assertIsNot(restored.meta, source.meta)
                    self.assertFalse(np.shares_memory(restored.meta["coords"], source.meta["coords"]))

    def test_copy_and_view_keep_existing_metadata_behavior(self):
        for field_class in FIELD_CLASSES:
            source = self.field(field_class)
            for copier in (lambda value: value.copy(), copy.copy, copy.deepcopy):
                with self.subTest(field=field_class.__name__, copier=copier):
                    copied = copier(source)
                    self.assert_field_equal(copied, source)
                    self.assertFalse(np.shares_memory(copied, source))
                    self.assert_field_equal(pickle.loads(pickle.dumps(copied)), source)
            view = source[:, :1]
            self.assertIs(view.meta, source.meta)
            self.assertTrue(np.shares_memory(view, source))

    def test_legacy_files_load_from_both_original_module_paths(self):
        for module_name, relative_path in SOURCES.items():
            package = ModuleType("libs")
            package.__path__ = []
            validation = ModuleType("libs.validation")
            validation.__path__ = []
            module = ModuleType(module_name)
            module.np = np
            package.validation = validation
            setattr(validation, module_name.rsplit(".", 1)[-1], module)
            modules = {"libs": package, "libs.validation": validation, module_name: module}

            with patch.dict(sys.modules, modules):
                old_node = metric_field_node(relative_path)
                old_node.body = [
                    node for node in old_node.body
                    if getattr(node, "name", None) not in {"__reduce__", "__setstate__"}
                ]
                exec(compile(ast.Module(body=[old_node], type_ignores=[]), relative_path, "exec"), module.__dict__)
                legacy = module.MetricField([1.0, np.nan, 3.0], lost_buoy_ids=np.array([1, 2, 3]))
                payloads = [pickle.dumps(legacy, protocol=protocol)
                            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)]

                current_node = metric_field_node(relative_path)
                exec(compile(ast.Module(body=[current_node], type_ignores=[]), relative_path, "exec"), module.__dict__)
                for protocol, payload in enumerate(payloads):
                    with self.subTest(module=module_name, protocol=protocol):
                        restored = pickle.loads(payload)
                        self.assertIs(type(restored), module.MetricField)
                        np.testing.assert_array_equal(restored, legacy)
                        # Old files never contained metadata, so it cannot be recovered.
                        self.assertEqual(restored.meta, {})
                        self.assertEqual(pickle.loads(pickle.dumps(restored)).meta, {})

    def test_legacy_unversioned_ndarray_state_and_missing_metadata(self):
        values = np.arange(6).reshape(2, 3)
        state = values.__reduce__()[2]
        for field_class in FIELD_CLASSES:
            with self.subTest(field=field_class.__name__):
                restored = field_class([])
                restored.__setstate__(state[1:])
                np.testing.assert_array_equal(restored, values)
                self.assertEqual(restored.meta, {})
                del restored.meta
                self.assertEqual(pickle.loads(pickle.dumps(restored)).meta, {})

    def test_validator_save_preserves_nested_result_metadata(self):
        path = ROOT / SOURCES["libs.validation.validator"]
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        validator = next(node for node in tree.body if getattr(node, "name", None) == "Validator")
        save_method = next(node for node in validator.body if getattr(node, "name", None) == "save")
        namespace = {"pickle": pickle}
        exec(compile(ast.Module(body=[save_method], type_ignores=[]), str(path), "exec"), namespace)
        when = date(2023, 1, 1)
        fields = [self.field(field_class).copy() for field_class in FIELD_CLASSES]
        results = {
            "identity": {
                (name,): {"RawFieldAggregator": {when: field}}
                for name, field in zip(("validator_field", "aggregator_field"), fields)
            },
        }
        instance = SimpleNamespace(results=results, processed_dates={when})
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "results.pkl"
            namespace["save"](instance, filename)
            with filename.open("rb") as handle:
                restored = pickle.load(handle)
        self.assertEqual(set(restored), {"results", "processed_dates"})
        self.assertEqual(restored["processed_dates"], {when})
        for name, expected in zip(("validator_field", "aggregator_field"), fields):
            self.assert_field_equal(
                restored["results"]["identity"][(name,)]["RawFieldAggregator"][when], expected,
            )


if __name__ == "__main__":
    unittest.main()
